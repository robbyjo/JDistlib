/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.IntegrationResult;
import jdistlib.rng.MersenneTwister;

public class NumericalRoadmapTest {
	@Test
	public void randomizedProbesAreBudgetedSeededAndReproducible() {
		long seed = 1234567L;
		int budget = 32;
		int exploratory = (budget + 1) / 2;
		double center = new Random(seed).nextDouble() / exploratory;
		FunctionAnalysisOptions settings = FunctionAnalysisOptions.builder()
				.sampleCount(9)
				.repeatabilityChecks(0)
				.randomizedProbeBudget(budget)
				.adaptiveProbeRounds(4)
				.randomSeed(seed)
				.build();
		FunctionAnalysis first = ProbabilityFunctionAnalyzer.analyze(
				x -> Math.abs(x - center) < 1e-14 ? -1.0 : 1.0,
				0.0, 1.0, settings);
		FunctionAnalysis second = ProbabilityFunctionAnalyzer.analyze(
				x -> Math.abs(x - center) < 1e-14 ? -1.0 : 1.0,
				0.0, 1.0, settings);

		assertTrue(hasCode(first, "NEGATIVE"));
		assertTrue(hasCode(first, "RANDOMIZED_PROBES"));
		assertEquals(9 + budget, first.getSampledPoints());
		assertEquals(budget, first.getRandomizedSampledPoints());
		assertEquals(seed, first.getRandomSeed());
		assertEquals(firstFindingX(first, "NEGATIVE"),
				firstFindingX(second, "NEGATIVE"), 0.0);
	}

	@Test
	public void arbitraryAbsoluteMomentsReportLeftAndRightConvergence() {
		NumericalContinuousDistribution uniform =
				new NumericalContinuousDistribution(x -> 1.0, -1.0, 1.0);
		DistributionAnalysis report = uniform.analyzeDistribution(
				MomentAnalysisOptions.builder().orders(0.5, 3.0, 4.0)
						.splitPoint(0.0).build());

		assertEquals(3, report.getAbsoluteMoments().size());
		for (double order : new double[] {0.5, 3.0, 4.0}) {
			AbsoluteMomentAnalysis moment = report.getAbsoluteMoment(order);
			assertNotNull(moment);
			assertTrue(moment.isStable());
			assertEquals(1.0 / (order + 1.0), moment.getValue(), 2e-9);
			assertEquals(0.5 / (order + 1.0), moment.getLeftValue(), 1e-9);
			assertEquals(0.5 / (order + 1.0), moment.getRightValue(), 1e-9);
		}
	}

	@Test
	public void constructionPoliciesControlAdvisoryFindings() {
		FunctionAnalysisOptions strict = FunctionAnalysisOptions.builder()
				.sampleCount(17).randomizedProbeBudget(0)
				.discontinuityRatio(100.0)
				.constructionPolicy(ConstructionPolicy.STRICT).build();
		FunctionAnalysisOptions warning = FunctionAnalysisOptions.builder()
				.sampleCount(17).randomizedProbeBudget(0)
				.discontinuityRatio(100.0)
				.constructionPolicy(ConstructionPolicy.WARNING).build();
		NumericalDistributionBuildResult rejected =
				NumericalContinuousDistribution.analyze(
						x -> x < 0.5 ? 1.0 : 1000.0, 0.0, 1.0, strict);
		NumericalDistributionBuildResult accepted =
				NumericalContinuousDistribution.analyze(
						x -> x < 0.5 ? 1.0 : 1000.0, 0.0, 1.0, warning);

		assertTrue(rejected.getAnalysis().hasWarnings());
		assertFalse(rejected.canBuild());
		assertTrue(accepted.canBuild());
		assertTrue(accepted.getAnalysis().isSuitableForConstruction(
				ConstructionPolicy.WARNING));
		assertFalse(accepted.getAnalysis().isSuitableForConstruction(
				ConstructionPolicy.STRICT));
	}

	@Test
	public void uniformRejectionEnvelopeProvidesFastOptionalSampling() {
		NumericalContinuousDistribution triangular =
				new NumericalContinuousDistribution(x -> 2.0 * x, 0.0, 1.0);
		triangular.setRandomEngine(new MersenneTwister(42L));
		triangular.configureUniformRejectionSampling(Math.log(2.0), 1000);
		double sum = 0.0;
		for (int i = 0; i < 5000; i++) sum += triangular.random();
		assertEquals(2.0 / 3.0, sum / 5000.0, 0.02);
		assertTrue(triangular.isRejectionSamplingConfigured());

		UniformRejectionEnvelope invalid = new UniformRejectionEnvelope(
				0.0, 1.0, Math.log(0.1));
		assertThrows(IllegalStateException.class,
				() -> triangular.random(invalid, 100));
		triangular.clearRejectionSampling();
		assertFalse(triangular.isRejectionSamplingConfigured());
	}

	@Test
	public void doubleExponentialQuadratureHandlesAllInfiniteShapes() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(1e-10, 1e-10)
				.maxEvaluations(500000)
				.method(IntegrationOptions.Method.DOUBLE_EXPONENTIAL)
				.tanhSinhMaxLevels(14)
				.build();
		IntegrationResult right = Integrate.integrate(x -> Math.exp(-x),
				0.0, Double.POSITIVE_INFINITY, options);
		IntegrationResult left = Integrate.integrate(x -> Math.exp(x),
				Double.NEGATIVE_INFINITY, 0.0, options);
		IntegrationResult whole = Integrate.integrate(x -> Math.exp(-x * x),
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, options);

		assertTrue(right.detailedMessage(), right.isSuccess());
		assertTrue(left.detailedMessage(), left.isSuccess());
		assertTrue(whole.detailedMessage(), whole.isSuccess());
		assertEquals(1.0, right.result, 2e-10);
		assertEquals(1.0, left.result, 2e-10);
		assertEquals(Math.sqrt(Math.PI), whole.result, 3e-10);
	}

	private static boolean hasCode(FunctionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (finding.getCode().equals(code)) return true;
		}
		return false;
	}

	private static double firstFindingX(FunctionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (finding.getCode().equals(code)) return finding.getX();
		}
		return Double.NaN;
	}
}
