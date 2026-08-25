/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.ImmutableIntegrationResult;
import jdistlib.math.IntegrationOptions;
import jdistlib.rng.MersenneTwister;

public class NumericalFeatureSetTest {
	@Test
	public void fluentBuildersCoverOrdinaryLogDiscreteAndMixedLaws() {
		NumericalContinuousDistribution continuous =
				NumericalContinuousDistribution.builder()
						.kernel(x -> 2.0 * x)
						.support(0.0, 1.0)
						.singularities(0.5)
						.diagnosticPreset(DiagnosticPreset.FAST)
						.cdfTable(CdfTableOptions.builder()
								.tolerance(1e-8).build())
						.build();
		assertEquals(0.25, continuous.cumulative(0.5), 2e-8);

		NumericalContinuousDistribution log =
				NumericalContinuousDistribution.builder()
						.logKernel(x -> -0.5 * x * x)
						.support(-5.0, 5.0)
						.withoutAnalysis()
						.build();
		assertEquals(0.0, log.mode(), 2e-7);

		NumericalDiscreteDistribution discrete =
				NumericalDiscreteDistribution.builder()
						.weights(x -> x + 1.0)
						.integerSupport(0, 2)
						.build();
		assertEquals(0.5, discrete.density(2.0, false), 0.0);

		NumericalSupport support = NumericalSupport.builder()
				.interval(0.0, 1.0).atom(2.0).build();
		NumericalPiecewiseDistribution mixed =
				NumericalPiecewiseDistribution.builder()
						.kernel(x -> 1.0)
						.atomWeights(x -> 1.0)
						.support(support)
						.integrationOptions(IntegrationOptions.builder()
								.tolerances(1e-10, 1e-10).build())
						.build();
		assertEquals(0.5, mixed.density(2.0, false), 2e-12);
	}

	@Test
	public void diagnosticPresetsScaleWorkAndRemainEditable() {
		FunctionAnalysisOptions fast = DiagnosticPreset.FAST.options();
		FunctionAnalysisOptions standard = FunctionAnalysisOptions.forPreset(
				DiagnosticPreset.STANDARD);
		FunctionAnalysisOptions thorough = DiagnosticPreset.THOROUGH.options();
		assertTrue(fast.getSampleCount() < standard.getSampleCount());
		assertTrue(standard.getSampleCount() < thorough.getSampleCount());
		assertTrue(fast.getRandomizedProbeBudget()
				< thorough.getRandomizedProbeBudget());
		assertEquals(17, fast.toBuilder().sampleCount(17).build().getSampleCount());
	}

	@Test
	public void compositionSupportsMixtureTruncationTransformAndCensoring() {
		NumericalContinuousDistribution uniform = uniform(0.0, 1.0);
		NumericalContinuousDistribution triangular =
				NumericalContinuousDistribution.builder().kernel(x -> 2.0 * x)
						.support(0.0, 1.0).withoutAnalysis().build();
		MixtureDistribution mixture = Distributions.mixture(
				new double[] {0.25, 0.75}, uniform, triangular);
		assertEquals(1.0, mixture.density(0.5, false), 2e-12);
		assertEquals(0.3125, mixture.cumulative(0.5), 2e-11);
		assertEquals(2.0 / 3.0, mixture.quantile(0.5), 2e-10);

		TruncatedContinuousDistribution truncated = Distributions.truncate(
				uniform, 0.2, 0.8);
		assertEquals(1.0 / 0.6, truncated.density(0.5, false), 2e-12);
		assertEquals(0.5, truncated.cumulative(0.5), 2e-12);
		assertEquals(0.35, truncated.quantile(0.25), 2e-11);

		MonotoneTransformDistribution transformed = Distributions.affine(
				uniform, 1.0, -2.0);
		assertEquals(-1.0, transformed.getLowerBound(), 0.0);
		assertEquals(1.0, transformed.getUpperBound(), 0.0);
		assertEquals(0.5, transformed.density(0.0, false), 2e-12);
		assertEquals(0.5, transformed.cumulative(0.0), 2e-12);

		CensoredDistribution censored = Distributions.censor(uniform, 0.2, 0.8);
		assertEquals(0.2, censored.getLowerAtomProbability(), 2e-12);
		assertEquals(0.2, censored.getUpperAtomProbability(), 2e-12);
		assertEquals(0.2, censored.density(0.2, false), 2e-12);
		assertEquals(0.2, censored.cumulative(0.2), 2e-12);
		assertEquals(0.8, censored.quantile(0.95), 2e-12);
		TruncatedContinuousDistribution upperPart = Distributions.truncate(
				uniform, 0.2, 1.0);
		MixtureDistribution atomMixture = Distributions.mixture(
				new double[] {0.5, 0.5, 0.0}, censored, upperPart,
				uniform(-100.0, 100.0));
		assertEquals(0.2, atomMixture.quantile(0.05), 0.0);
		assertEquals(0.2, atomMixture.getLowerBound(), 0.0);
		assertEquals(1.0, atomMixture.getUpperBound(), 0.0);
	}

	@Test
	public void aliasAndAdaptiveRejectionSamplingReportTheirStrategies() {
		NumericalDiscreteDistribution discrete =
				new NumericalDiscreteDistribution(x -> x == 0.0 ? 1.0 : 3.0, 0, 1);
		discrete.setRandomEngine(new MersenneTwister(123L));
		int ones = 0;
		for (int i = 0; i < 20000; i++) if (discrete.random() == 1.0) ones++;
		assertEquals(0.75, ones / 20000.0, 0.015);
		assertEquals(SamplingStrategy.WALKER_ALIAS, discrete.getSamplingStrategy());

		NumericalContinuousDistribution normal =
				NumericalContinuousDistribution.builder()
						.logKernel(x -> -0.5 * x * x)
						.support(-4.0, 4.0)
						.withoutAnalysis()
						.adaptiveRejectionSampling(x -> -x, -2.0, 0.0, 2.0)
						.build();
		normal.setRandomEngine(new MersenneTwister(456L));
		double sum = 0.0;
		double squared = 0.0;
		for (int i = 0; i < 10000; i++) {
			double value = normal.random();
			sum += value;
			squared += value * value;
		}
		assertEquals(0.0, sum / 10000.0, 0.04);
		assertEquals(1.0, squared / 10000.0, 0.06);
		assertEquals(SamplingStrategy.ADAPTIVE_LOG_CONCAVE_REJECTION,
				normal.getSamplingStrategy());
		assertTrue(normal.getSamplingStrategyExplanation().contains("tangent"));
	}

	@Test
	public void tailHelpersCertifyGeometricPowerAndFinitePrefixRemainders() {
		CertifiedDiscreteOptions tight = CertifiedDiscreteOptions.builder()
				.minimumTerms(4).maximumTerms(10000)
				.omittedProbabilityTolerance(1e-8).build();
		CertifiedInfiniteDiscreteDistribution geometric =
				CertifiedInfiniteDiscreteDistribution.rightInfinite(
						x -> Math.pow(0.5, x), 0,
						DiscreteTailBounds.geometricRatio(0.5), tight);
		assertTrue(geometric.getOmittedProbabilityUpperBound() <= 1e-8);

		CertifiedInfiniteDiscreteDistribution power =
				CertifiedInfiniteDiscreteDistribution.rightInfinite(
						x -> 1.0 / ((x + 1.0) * (x + 1.0)), 0,
						DiscreteTailBounds.rightPowerLaw(2.0, 1.0),
						CertifiedDiscreteOptions.builder().minimumTerms(4)
								.maximumTerms(10000)
								.omittedProbabilityTolerance(1e-3).build());
		assertTrue(power.getOmittedProbabilityUpperBound() <= 1e-3);

		DiscreteTailBound delayed = DiscreteTailBounds.afterFinitePrefix(10,
				DiscreteTailBounds.geometricRatio(0.5));
		assertEquals(Double.MAX_VALUE, delayed.upperBound(9, 1e-3), 0.0);
		assertEquals(2e-3, delayed.upperBound(10, 1e-3), 0.0);
		CertifiedInfiniteDiscreteDistribution prefixed =
				CertifiedInfiniteDiscreteDistribution.rightInfinite(
						x -> Math.pow(0.5, x), 0, delayed, tight);
		assertTrue(prefixed.getIncludedTerms() >= 10);
	}

	@Test
	public void summariesCoverExpectationsMomentsEntropyModesAndIntervals() {
		NumericalContinuousDistribution uniform = uniform(0.0, 1.0);
		ImmutableIntegrationResult expectation = uniform.expectation(x -> x * x);
		assertTrue(expectation.detailedMessage(), expectation.isSuccess());
		assertEquals(1.0 / 3.0, expectation.getValue(), 2e-11);
		assertEquals(1.0 / 3.0, uniform.rawMoment(2.0).getValue(), 2e-11);
		assertEquals(1.0 / 12.0, uniform.centralMoment(2.0).getValue(), 2e-11);
		assertEquals(0.0, uniform.entropy().getValue(), 2e-11);
		ProbabilityInterval interval = uniform.probabilityInterval(0.9);
		assertEquals(0.05, interval.getLower(), 2e-10);
		assertEquals(0.95, interval.getUpper(), 2e-10);

		NumericalDiscreteDistribution discrete =
				new NumericalDiscreteDistribution(x -> x + 1.0, 0, 2);
		assertEquals(4.0 / 3.0, discrete.expectation(x -> x), 2e-15);
		assertEquals(7.0 / 3.0, discrete.rawMoment(2.0), 2e-15);
		assertEquals(5.0 / 9.0, discrete.centralMoment(2.0), 2e-15);
		assertEquals(2.0, discrete.mode(), 0.0);
		assertTrue(discrete.entropy() > 0.0);
	}

	@Test
	public void batchApisReuseCachesAndCallerOwnedStorage() {
		NumericalContinuousDistribution uniform = uniform(0.0, 1.0);
		double[] input = {0.1, 0.25, 0.75, 0.9};
		double[] output = new double[6];
		uniform.cumulativeInto(input, 1, output, 2, 2, true, false);
		assertEquals(0.25, output[2], 2e-9);
		assertEquals(0.75, output[3], 2e-9);
		assertTrue(uniform.getCdfTable().size() > 0);

		uniform.quantileInto(new double[] {0.2, 0.8}, 0, output, 0, 2,
				true, false);
		assertEquals(0.2, output[0], 2e-10);
		assertEquals(0.8, output[1], 2e-10);
		uniform.randomInto(output, 1, 3);
		for (int i = 1; i < 4; i++) assertTrue(output[i] >= 0.0 && output[i] <= 1.0);
		double[] overlapping = {0.25, 0.75, -1.0};
		uniform.cumulativeInto(overlapping, 0, overlapping, 1, 2, true, false);
		assertEquals(0.25, overlapping[1], 2e-9);
		assertEquals(0.75, overlapping[2], 2e-9);
		assertThrows(IllegalArgumentException.class,
				() -> uniform.randomInto(output, 5, 2));
	}

	private static NumericalContinuousDistribution uniform(double lower,
			double upper) {
		return NumericalContinuousDistribution.builder().kernel(x -> 1.0)
				.support(lower, upper).withoutAnalysis().build();
	}
}
