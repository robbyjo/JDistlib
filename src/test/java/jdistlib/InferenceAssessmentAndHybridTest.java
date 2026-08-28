/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Constraints;
import jdistlib.inference.ContinuousBlockMetropolisKernel;
import jdistlib.inference.CoordinateSupport;
import jdistlib.inference.FiniteDiscreteGibbsKernel;
import jdistlib.inference.HybridSampler;
import jdistlib.inference.HybridSamplingResult;
import jdistlib.inference.LooModelComparison;
import jdistlib.inference.MixedStateSpace;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ObservationMetadata;
import jdistlib.inference.PointwiseLogLikelihoodDraws;
import jdistlib.inference.PredictiveStacking;
import jdistlib.inference.ProjectionPredictiveSelection;
import jdistlib.inference.PsisLoo;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.ShrinkageSelection;
import jdistlib.inference.Waic;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

public class InferenceAssessmentAndHybridTest {
	@Test public void pointwiseContractWorksForBuilderAndCompiledModels() {
		BayesianModel built = new ModelBuilder().parameter("mu", Constraints.real(), 0.0)
				.factor("prior", new String[] {"mu"}, state -> -0.5 * state.scalar("mu") * state.scalar("mu"))
				.likelihood("y[1]", "y", new String[] {"mu"}, state -> -0.5 * square(1.0 - state.scalar("mu")))
				.likelihood("y[2]", "y", new String[] {"mu"}, state -> -0.5 * square(2.0 - state.scalar("mu"))).build();
		assertArrayEquals(new double[] {-0.5, -2.0}, built.pointwiseLogLikelihood(new double[] {0.0}), 0.0);
		assertEquals("y", built.observationMetadata().group(1));

		String source = "data { int<lower=1> N; array[N] real y; } parameters { real mu; } "
				+ "model { mu ~ normal(0,1); y ~ normal(mu,1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {2.0}); data.put("y", new double[] {1.0, 2.0});
		BayesianModel compiled = ModelScript.compileStan(source, data).model();
		assertEquals(2, compiled.observationMetadata().size());
		assertArrayEquals(new String[] {"y[1]", "y[2]"}, compiled.observationMetadata().names());
		double[] likelihood = compiled.pointwiseLogLikelihood(new double[] {0.0});
		assertEquals(-0.5 * Math.log(2.0 * Math.PI) - 0.5, likelihood[0], 1e-12);
		assertEquals(-0.5 * Math.log(2.0 * Math.PI) - 2.0, likelihood[1], 1e-12);
	}

	@Test public void looWaicComparisonAndStackingUsePairedPointwiseValues() {
		ObservationMetadata metadata = ObservationMetadata.ungrouped("a", "b", "c");
		double[][] first = new double[40][3], second = new double[40][3];
		for (int draw = 0; draw < first.length; draw++) for (int observation = 0; observation < 3; observation++) {
			first[draw][observation] = -0.8 + 0.03 * ((draw + observation) % 5);
			second[draw][observation] = first[draw][observation] - 1.0;
		}
		PointwiseLogLikelihoodDraws firstDraws = new PointwiseLogLikelihoodDraws(metadata, first, new int[] {0, 40});
		PointwiseLogLikelihoodDraws secondDraws = new PointwiseLogLikelihoodDraws(metadata, second, new int[] {0, 40});
		PsisLoo.Result firstLoo = PsisLoo.compute(firstDraws), secondLoo = PsisLoo.compute(secondDraws);
		assertTrue(firstLoo.elpd() > secondLoo.elpd()); assertEquals(3, firstLoo.paretoK().length);
		Waic.Result waic = Waic.compute(firstDraws); assertTrue(Double.isFinite(waic.waic())); assertTrue(waic.reliable());
		assertEquals("first", LooModelComparison.compare(LooModelComparison.model("second", secondLoo),
				LooModelComparison.model("first", firstLoo)).get(0).name());
		PredictiveStacking.Result stacking = PredictiveStacking.fit(new String[] {"first", "second"}, firstLoo, secondLoo);
		assertTrue(stacking.weights()[0] > 0.99); assertEquals(1.0, stacking.weights()[0] + stacking.weights()[1], 1e-12);
	}

	@Test public void looUsesExplicitFallbackForUnstableImportanceRatios() {
		double[][] values = new double[20][1];
		for (int draw = 0; draw < values.length; draw++) values[draw][0] = draw < 15 ? 0.0 : -Math.pow(2.0, draw - 15);
		PointwiseLogLikelihoodDraws draws = new PointwiseLogLikelihoodDraws(
				ObservationMetadata.ungrouped("outlier"), values, new int[] {0, 20});
		PsisLoo.Result raw = PsisLoo.compute(draws);
		assertFalse(raw.reliable()); assertTrue(raw.paretoK()[0] > 0.7);
		PsisLoo.Result replaced = PsisLoo.compute(draws, (observation, name) -> -3.0);
		assertTrue(replaced.reliable()); assertTrue(replaced.fallbackUsed()[0]); assertEquals(-3.0, replaced.elpd(), 0.0);
	}

	@Test public void projectionAndShrinkageSelectionRecoverTheRelevantVariable() {
		double[][] design = {{-2, 1}, {-1, -1}, {0, 1}, {1, -1}, {2, 1}};
		double[][] coefficients = new double[20][2]; double[] scales = new double[20];
		for (int draw = 0; draw < coefficients.length; draw++) { coefficients[draw][0] = 2.0 + 0.01 * draw; coefficients[draw][1] = 0.001 * (draw - 10); scales[draw] = 1.0; }
		ProjectionPredictiveSelection.Result projection = ProjectionPredictiveSelection.select(design,
				new String[] {"signal", "noise"}, coefficients, null, scales, 0.001);
		assertEquals("signal", projection.path().get(1).variableNames()[0]);
		assertEquals(1, projection.selectedModel().size());
		ShrinkageSelection.Result shrinkage = ShrinkageSelection.analyze(new String[] {"signal", "noise"}, coefficients, 0.1, 0.9);
		assertEquals("signal", shrinkage.ranking().get(0).name()); assertEquals(1, shrinkage.selected().size());
	}

	@Test public void hybridSamplerRespectsTypedSupportsAndReportsKernelDiagnostics() {
		MixedStateSpace space = new MixedStateSpace(CoordinateSupport.binary(), CoordinateSupport.real());
		HybridSampler sampler = new HybridSampler(space, new FiniteDiscreteGibbsKernel(0),
				new ContinuousBlockMetropolisKernel(new int[] {1}, 0.8));
		HybridSamplingResult result = sampler.sampleMixed(state -> {
			double mean = state[0] == 0.0 ? -1.0 : 1.0; return -0.5 * square(state[1] - mean);
		}, new double[] {0.0, 0.0}, SamplingOptions.builder().warmupIterations(100).sampleIterations(400)
				.thinning(1).stepSize(0.8).build(), new MersenneTwister(811));
		assertEquals(400, result.chain().size()); assertEquals(2, result.diagnostics().kernelCount());
		assertTrue(result.diagnostics().acceptanceRate(1) > 0.1); assertTrue(result.diagnostics().acceptanceRate(1) < 1.0);
		boolean sawZero = false, sawOne = false;
		for (double[] draw : result.chain().samples()) {
			assertTrue(space.contains(draw)); sawZero |= draw[0] == 0.0; sawOne |= draw[0] == 1.0;
		}
		assertTrue(sawZero && sawOne); assertFalse(result.chain().warnings().size() > 0);
	}
	private static double square(double value) { return value * value; }
}
