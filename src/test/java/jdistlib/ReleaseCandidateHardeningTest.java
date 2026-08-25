/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.TimeUnit;

import org.junit.Test;

import jdistlib.math.IntegrationOptions;
import jdistlib.rng.MersenneTwister;

/** Adversarial release-candidate coverage for user-supplied distributions. */
public class ReleaseCandidateHardeningTest {
	@Test
	public void logKernelAnalysisPreservesExtremeDynamicRange() {
		FunctionAnalysisOptions options = DiagnosticPreset.FAST.options().toBuilder()
				.dynamicRangeOrders(5.0)
				.constructionPolicy(ConstructionPolicy.WARNING)
				.build();
		NumericalDistributionBuildResult candidate =
				NumericalContinuousDistribution.analyzeLogKernel(
						x -> -1000.0 * x * x, -1.0, 1.0, options);
		assertTrue(hasFinding(candidate.getAnalysis(), "DYNAMIC_RANGE"));
		assertTrue(candidate.canBuild());
		NumericalContinuousDistribution distribution = candidate.build();
		assertTrue(Double.isFinite(distribution.density(0.0, true)));
		assertEquals(0.5, distribution.cumulative(0.0), 2e-8);
	}

	@Test
	public void logKernelBuilderRejectsInvalidValuesAndAcceptsZeroMass() {
		assertThrows(IllegalStateException.class, () ->
				NumericalContinuousDistribution.builder()
						.logKernel(x -> x < 0.0 ? Double.NaN : 0.0)
						.support(-1.0, 1.0)
						.diagnosticPreset(DiagnosticPreset.FAST)
						.build());
		assertThrows(IllegalStateException.class, () ->
				NumericalContinuousDistribution.builder()
						.logKernel(x -> Double.POSITIVE_INFINITY)
						.support(-1.0, 1.0)
						.diagnosticPreset(DiagnosticPreset.FAST)
						.build());

		NumericalContinuousDistribution halfUniform =
				NumericalContinuousDistribution.builder()
						.logKernel(x -> x < 0.0
								? Double.NEGATIVE_INFINITY : 0.0)
						.support(-1.0, 1.0)
						.diagnosticPreset(DiagnosticPreset.FAST)
						.build();
		assertEquals(0.0, halfUniform.density(-0.5, false), 0.0);
		assertEquals(1.0, halfUniform.density(0.5, false), 2e-10);
	}

	@Test
	public void logKernelDiagnosticsHonorCallbackTimeouts() {
		IntegrationOptions integration = IntegrationOptions.builder()
				.callbackExecution(IntegrationOptions.CallbackExecution.ISOLATED_DAEMON)
				.maxCallbackTime(5L, TimeUnit.MILLISECONDS)
				.maxTotalTime(100L, TimeUnit.MILLISECONDS)
				.build();
		FunctionAnalysis analysis = ProbabilityFunctionAnalyzer.analyzeLogKernel(x -> {
			try {
				Thread.sleep(10000L);
			} catch (InterruptedException exception) {
				Thread.currentThread().interrupt();
			}
			return 0.0;
		}, 0.0, 1.0, FunctionAnalysisOptions.builder()
				.sampleCount(9).randomizedProbeBudget(0).repeatabilityChecks(0)
				.integrationOptions(integration).build());
		assertTrue(hasFinding(analysis, "CALLBACK_TIME_LIMIT"));
		assertTrue(analysis.hasErrors());
	}

	@Test
	public void compositionHandlesEndpointAtomsAndDecreasingMaps() {
		NumericalContinuousDistribution uniform =
				NumericalContinuousDistribution.builder().kernel(x -> 1.0)
						.support(0.0, 1.0).withoutAnalysis().build();
		CensoredDistribution censored = Distributions.censor(uniform, 0.2, 0.8);
		MonotoneTransformDistribution reflected = Distributions.affine(
				censored, 1.0, -2.0);
		assertEquals(0.2, reflected.density(-0.6, false), 2e-12);
		assertEquals(0.2, reflected.cumulative(-0.6), 2e-12);
		assertEquals(0.6, reflected.quantile(0.95), 2e-12);

		MixtureDistribution mixture = Distributions.mixture(
				new double[] {1.0, 0.0}, censored,
				NumericalContinuousDistribution.builder().kernel(x -> 1.0)
						.support(-1e6, 1e6).withoutAnalysis().build());
		assertEquals(0.2, mixture.getLowerBound(), 0.0);
		assertEquals(0.2, mixture.quantile(0.1), 0.0);
	}

	@Test
	public void seededDiscreteSamplingIsExactlyRepeatable() {
		NumericalDiscreteDistribution first =
				new NumericalDiscreteDistribution(x -> x + 1.0, 0, 4);
		NumericalDiscreteDistribution second =
				new NumericalDiscreteDistribution(x -> x + 1.0, 0, 4);
		first.setRandomEngine(new MersenneTwister(20260825L));
		second.setRandomEngine(new MersenneTwister(20260825L));
		assertArrayEquals(first.random(1000), second.random(1000), 0.0);
		assertFalse(Double.isNaN(first.entropy()));
	}

	private static boolean hasFinding(FunctionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (code.equals(finding.getCode())) return true;
		}
		return false;
	}
}
