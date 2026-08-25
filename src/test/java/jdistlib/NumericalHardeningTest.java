/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.IntegrationResult;
import jdistlib.math.IntegrationStabilityResult;

public class NumericalHardeningTest {
	@Test
	public void hardenedIntegrationReportsCallbackFailuresAndBudgets() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(1e-12, 1e-12).maxEvaluations(1000).build();
		IntegrationResult callback = Integrate.integrate(x -> {
			if (x > 0.4) throw new IllegalStateException("deliberate");
			return x;
		}, 0.0, 1.0, options);
		assertEquals(7, callback.ier);
		assertTrue(callback.failureX > 0.4);
		assertNotNull(callback.cause);
		assertTrue(callback.detailedMessage().contains("deliberate"));

		IntegrationResult nonFinite = Integrate.integrate(
				x -> x == 0.5 ? Double.NaN : x, 0.0, 1.0, options);
		assertEquals(10, nonFinite.ier);
		assertEquals(0.5, nonFinite.failureX, 0.0);

		IntegrationResult budget = Integrate.integrate(x -> Math.exp(x), 0.0, 1.0,
				options.toBuilder().maxEvaluations(5).build());
		assertEquals(9, budget.ier);
		assertTrue(budget.neval <= 5);

		IntegrationResult cancelled = Integrate.integrate(x -> x, 0.0, 1.0,
				options.toBuilder().cancellation(() -> true).build());
		assertEquals(8, cancelled.ier);
	}

	@Test
	public void breakpointsHandleAnInteriorIntegrableSingularity() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(1e-10, 1e-10)
				.subdivisions(300)
				.breakpoints(0.5)
				.build();
		IntegrationResult result = Integrate.integrate(
				x -> 1.0 / Math.sqrt(Math.abs(x - 0.5)), 0.0, 1.0, options);
		assertTrue(result.detailedMessage(), result.isSuccess());
		assertEquals(2.0 * Math.sqrt(2.0), result.result, 2e-9);
	}

	@Test
	public void tanhSinhHandlesFiniteEndpointSingularities() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(3e-8, 3e-8)
				.maxEvaluations(100000)
				.method(IntegrationOptions.Method.TANH_SINH)
				.tanhSinhMaxLevels(14)
				.build();
		IntegrationResult result = Integrate.integrate(
				x -> 1.0 / Math.sqrt(x * (1.0 - x)), 0.0, 1.0, options);
		assertTrue(result.detailedMessage(), result.isSuccess());
		assertEquals(Math.PI, result.result, 5e-8);
	}

	@Test
	public void stabilityAssessmentRepeatsWithTighterAndSplitSettings() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(1e-9, 1e-9).subdivisions(100).build();
		IntegrationStabilityResult stability = Integrate.assessStability(
				x -> x * x, 0.0, 1.0, options);
		assertTrue(stability.message(), stability.isStable());
		assertEquals(1.0 / 3.0, stability.getTightened().result, 2e-14);
		assertEquals(1.0 / 3.0, stability.getSplit().result, 2e-14);

		IntegrationResult hugeInterval = Integrate.integrate(x -> 1.0,
				1.0e308, 1.1e308, options);
		assertTrue(hugeInterval.detailedMessage(), hugeInterval.isSuccess());
		assertEquals(1.0e307, hugeInterval.result, 2e293);
	}

	@Test
	public void functionAnalysisFindsInvalidAndNonDeterministicKernels() {
		FunctionAnalysis negative = ProbabilityFunctionAnalyzer.analyze(
				x -> x - 0.25, 0.0, 1.0);
		assertTrue(negative.hasErrors());
		assertTrue(hasCode(negative, "NEGATIVE"));

		AtomicInteger calls = new AtomicInteger();
		FunctionAnalysis changing = ProbabilityFunctionAnalyzer.analyze(
				x -> 1.0 + calls.incrementAndGet() * 1e-3, 0.0, 1.0,
				FunctionAnalysisOptions.builder().sampleCount(17)
						.repeatabilityChecks(3).build());
		assertTrue(changing.hasErrors());
		assertTrue(hasCode(changing, "NON_DETERMINISTIC"));
	}

	@Test
	public void analysisCanConstructAndRetainsFailureReports() {
		NumericalDistributionBuildResult good =
				NumericalContinuousDistribution.analyze(
						x -> Math.exp(-x * x),
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
		assertTrue(good.getAnalysis().isSuitableForConstruction());
		assertTrue(good.canBuild());
		assertEquals(Math.sqrt(Math.PI),
				good.build().getNormalizationConstant(), 2e-9);

		NumericalDistributionBuildResult bad =
				NumericalContinuousDistribution.analyze(x -> -1.0, 0.0, 1.0,
						FunctionAnalysisOptions.builder().sampleCount(17).build());
		assertFalse(bad.canBuild());
		assertNotNull(bad.getFailure());
		assertThrows(IllegalStateException.class, bad::build);
	}

	@Test
	public void logKernelAvoidsOrdinaryScaleOverflow() {
		NumericalContinuousDistribution distribution =
				NumericalContinuousDistribution.fromLogKernel(
						x -> 1000.0 - 0.5 * x * x,
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
		assertTrue(Double.isInfinite(distribution.getNormalizationConstant()));
		assertEquals(1000.0 + 0.5 * Math.log(2.0 * Math.PI),
				distribution.getLogNormalizationConstant(), 2e-10);
		for (double x : new double[] {-2.0, 0.0, 1.5}) {
			assertEquals(Normal.density(x, 0.0, 1.0, true),
					distribution.density(x, true), 2e-10);
		}

		NumericalDiscreteDistribution discrete =
				NumericalDiscreteDistribution.fromLogWeights(
						x -> 1000.0 + x * Math.log(2.0), 0, 2);
		assertTrue(Double.isInfinite(discrete.getNormalizationConstant()));
		assertEquals(1000.0 + Math.log(7.0),
				discrete.getLogNormalizationConstant(), 2e-13);
		assertEquals(4.0 / 7.0, discrete.density(2.0, false), 2e-14);
	}

	@Test
	public void distributionAnalysisChecksTailsInversesAndMoments() {
		NumericalContinuousDistribution continuous =
				new NumericalContinuousDistribution(x -> x, 0.0, 1.0,
						IntegrationOptions.builder().tolerances(1e-11, 1e-11)
								.subdivisions(200).build());
		DistributionAnalysis continuousReport = continuous.analyzeDistribution();
		assertFalse(continuousReport.hasErrors());
		assertTrue(continuousReport.areMomentsStable());
		assertEquals(2.0 / 3.0, continuousReport.getMean(), 2e-9);
		assertEquals(1.0 / 18.0, continuousReport.getVariance(), 2e-9);
		assertTrue(continuousReport.getMaximumTailDisagreement() < 1e-9);

		NumericalDiscreteDistribution discrete =
				new NumericalDiscreteDistribution(x -> x + 1.0, 0, 3);
		DistributionAnalysis discreteReport = discrete.analyzeDistribution();
		assertFalse(discreteReport.hasErrors());
		assertTrue(discreteReport.areMomentsStable());
		assertEquals(2.0, discreteReport.getMean(), 2e-15);
		assertEquals(1.0, discreteReport.getVariance(), 2e-15);
	}

	private static boolean hasCode(FunctionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (finding.getCode().equals(code)) return true;
		}
		return false;
	}
}
