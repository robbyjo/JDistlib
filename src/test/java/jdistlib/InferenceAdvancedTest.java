/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.ComponentWiseMetropolis;
import jdistlib.inference.Constraints;
import jdistlib.inference.DifferentiableModelFactor;
import jdistlib.inference.GibbsSampler;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.HamiltonianMonteCarlo;
import jdistlib.inference.LogDensity;
import jdistlib.inference.DifferentiableLogDensity;
import jdistlib.inference.IterationStats;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.WarmupResult;
import jdistlib.inference.MetropolisBlockKernel;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelEvaluationCache;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.ParameterConstraint;
import jdistlib.inference.RandomWalkMetropolis;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.SliceSampler;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.inference.lang.ModelScriptException;
import jdistlib.rng.MersenneTwister;

public class InferenceAdvancedTest {
	@Test public void orderedAndSimplexTransformsRoundTripAndDifferentiate() {
		ParameterConstraint ordered = Constraints.ordered(3);
		double[] orderedValue = {-2.0, 0.5, 4.0};
		double[] orderedFree = new double[3];
		double[] orderedRoundTrip = new double[3];
		ordered.unconstrain(orderedValue, 0, orderedFree, 0);
		ordered.constrain(orderedFree, 0, orderedRoundTrip, 0);
		assertArrayEquals(orderedValue, orderedRoundTrip, 1e-14);

		ParameterConstraint simplex = Constraints.simplex(4);
		double[] simplexValue = {0.1, 0.2, 0.3, 0.4};
		double[] simplexFree = new double[3];
		double[] simplexRoundTrip = new double[4];
		simplex.unconstrain(simplexValue, 0, simplexFree, 0);
		simplex.constrain(simplexFree, 0, simplexRoundTrip, 0);
		assertArrayEquals(simplexValue, simplexRoundTrip, 2e-15);

		BayesianModel model = new ModelBuilder()
				.parameter("weights", simplex, simplexValue)
				.factor("smooth simplex", new String[] {"weights"}, state -> {
					double result = 0.0;
					for (double value : state.vector("weights")) result -= value * value;
					return result;
				}).build();
		assertFalse(model.hasAnalyticGradient());
		GradientCheckResult check = Gradients.check(model, model.initialState(), 1e-5, 1e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void analyticSimplexPullbackMatchesFiniteDifferences() {
		BayesianModel model = new ModelBuilder()
				.parameter("weights", Constraints.simplex(4), 0.1, 0.2, 0.3, 0.4)
				.factor("weighted objective", new String[] {"weights"},
						new DifferentiableModelFactor() {
					@Override public double logDensityAndAddGradient(
							jdistlib.inference.ModelState state, double[] gradient) {
						double result = 0.0;
						for (int i = 0; i < 4; i++) {
							double value = state.value("weights", i);
							result -= (i + 1.0) * value * value;
							state.addGradient("weights", i, -2.0 * (i + 1.0) * value, gradient);
						}
						return result;
					}
				}).build();
		assertTrue(model.hasAnalyticGradient());
		GradientCheckResult check = Gradients.check(model, model.initialState(), 2e-5, 2e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void scriptDependenciesExpandThroughTransformsAndIndexing() {
		String source = "data { vector[2] y; } parameters { real mu; } "
				+ "transformed parameters { real twice = 2 * mu; } "
				+ "model { y[1] ~ normal(twice, 1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("y", new double[] {1.0, 1000.0});
		CompiledModelScript compiled = ModelScript.compile(source, data);
		assertTrue(compiled.model().graph().edges().stream()
				.anyMatch(edge -> edge.from().equals("parameter:mu")));
		double atZero = compiled.model().logDensity(new double[] {0.0});
		assertEquals(Normal.density(1.0, 0.0, 1.0, true), atZero, 1e-14);
		assertTrue(Gradients.check(compiled.model(), new double[] {0.3}, 2e-6, 2e-6).passed());
	}

	@Test(expected = ModelScriptException.class)
	public void scriptDataBoundsAreValidated() {
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("n", new double[] {-1});
		ModelScript.compile("data { int<lower=0> n; } parameters { real x; } model { x ~ normal(0,1); }", data);
	}

	@Test public void hmcSliceAndComponentSamplersHandleHeavyAndGaussianTargets() {
		BayesianModel normal = new ModelBuilder().parameter("x", Constraints.real(), 0.0)
				.factor("normal", new String[] {"x"}, ModelFactors.normalPrior("x", 0, 1)).build();
		SamplingOptions shortRun = SamplingOptions.builder().warmupIterations(100)
				.sampleIterations(160).leapfrogSteps(8).build();
		ChainResult hmc = new HamiltonianMonteCarlo().sample(normal, normal.initialState(),
				shortRun, new MersenneTwister(10));
		assertEquals(ChainResult.Status.SUCCESS, hmc.status());
		assertEquals(160, hmc.size());

		LogDensity cauchy = x -> -Math.log(Math.PI) - Math.log1p(x[0] * x[0]);
		ChainResult slice = new SliceSampler().sample(cauchy, new double[] {0.0},
				SamplingOptions.builder().warmupIterations(50).sampleIterations(100).build(),
				new MersenneTwister(11));
		assertEquals(ChainResult.Status.SUCCESS, slice.status());
		ChainResult component = new ComponentWiseMetropolis().sample(cauchy,
				new double[] {0.0}, SamplingOptions.builder().warmupIterations(50)
						.sampleIterations(100).build(), new MersenneTwister(12));
		assertEquals(ChainResult.Status.SUCCESS, component.status());
	}

	@Test public void fixedStepIsHonoredAndDenseMetricIsReported() {
		BayesianModel normal = new ModelBuilder().parameter("x", Constraints.real(), 0.0)
				.factor("normal", new String[] {"x"}, ModelFactors.normalPrior("x", 0, 1)).build();
		SamplingOptions fixed = SamplingOptions.builder().warmupIterations(0)
				.sampleIterations(12).adaptStepSize(false).adaptMassMatrix(false)
				.stepSize(0.03125).build();
		ChainResult hmc = new HamiltonianMonteCarlo().sample(normal,
				normal.initialState(), fixed, new MersenneTwister(55));
		ChainResult nuts = new NoUTurnSampler().sample(normal,
				normal.initialState(), fixed, new MersenneTwister(56));
		assertEquals(0.03125, hmc.warmup().finalStepSize(), 0.0);
		assertEquals(0.03125, nuts.warmup().finalStepSize(), 0.0);

		DifferentiableLogDensity correlated = new DifferentiableLogDensity() {
			@Override public double logDensityAndGradient(double[] x, double[] gradient) {
				double rho = 0.8, denominator = 1.0 - rho * rho;
				gradient[0] = -(x[0] - rho * x[1]) / denominator;
				gradient[1] = -(x[1] - rho * x[0]) / denominator;
				return -(x[0] * x[0] - 2.0 * rho * x[0] * x[1]
						+ x[1] * x[1]) / (2.0 * denominator);
			}
		};
		ChainResult dense = new NoUTurnSampler().sample(correlated, new double[] {0, 0},
				SamplingOptions.builder().warmupIterations(160).sampleIterations(20)
						.denseMassMatrix(true).build(), new MersenneTwister(57));
		double[][] inverse = dense.warmup().inverseMassMatrix();
		assertEquals(2, inverse.length);
		assertTrue(Double.isFinite(inverse[0][1]));
		assertTrue(Math.abs(inverse[0][1]) > 1e-6);
	}

	@Test public void mixedBlockGibbsAndCancellationAreExplicit() {
		LogDensity mixed = x -> {
			if (x[0] != 0.0 && x[0] != 1.0) return Double.NEGATIVE_INFINITY;
			double mean = x[0] == 0.0 ? -1.0 : 1.0;
			return -0.5 * (x[1] - mean) * (x[1] - mean);
		};
		GibbsSampler sampler = new GibbsSampler(
				(state, target, random) -> state[0] = random.nextDouble() < 0.5 ? 0.0 : 1.0,
				new MetropolisBlockKernel(new int[] {1}, 0.6, false));
		ChainResult result = sampler.sample(mixed, new double[] {0, 0},
				SamplingOptions.builder().warmupIterations(20).sampleIterations(80).build(),
				new MersenneTwister(22));
		assertEquals(ChainResult.Status.SUCCESS, result.status());
		for (double[] sample : result.samples()) assertTrue(sample[0] == 0.0 || sample[0] == 1.0);

		final AtomicInteger checks = new AtomicInteger();
		SamplingOptions cancelled = SamplingOptions.builder().warmupIterations(0)
				.sampleIterations(100).cancellation(() -> checks.incrementAndGet() > 5).build();
		ChainResult stopped = new RandomWalkMetropolis().sample(mixed,
				new double[] {0, 0}, cancelled, new MersenneTwister(23));
		assertEquals(ChainResult.Status.CANCELLED, stopped.status());
	}

	@Test public void checkpointContinuationPreservesExactStream() {
		LogDensity target = x -> -0.5 * x[0] * x[0];
		SamplingOptions twenty = SamplingOptions.builder().warmupIterations(0)
				.sampleIterations(20).adaptStepSize(false).stepSize(0.4).build();
		SamplingOptions forty = SamplingOptions.builder().warmupIterations(0)
				.sampleIterations(40).adaptStepSize(false).stepSize(0.4).build();
		RandomWalkMetropolis sampler = new RandomWalkMetropolis();
		ChainResult first = sampler.sample(target, new double[] {0}, twenty,
				new MersenneTwister(99));
		ChainResult continued = Chains.resume(sampler, target, first.checkpoint(), twenty);
		ChainResult direct = sampler.sample(target, new double[] {0}, forty,
				new MersenneTwister(99));
		for (int i = 0; i < twenty.sampleIterations(); i++)
			assertArrayEquals(direct.sample(i + 20), continued.sample(i), 0.0);
	}

	@Test public void factorCacheReevaluatesOnlyAffectedContractCorrectly() {
		BayesianModel model = new ModelBuilder()
				.parameter("x", Constraints.real(), 0.0)
				.parameter("y", Constraints.real(), 0.0)
				.factor("x", new String[] {"x"}, state -> -state.scalar("x") * state.scalar("x"))
				.factor("y", new String[] {"y"}, state -> -2.0 * state.scalar("y") * state.scalar("y"))
				.build();
		ModelEvaluationCache cache = new ModelEvaluationCache(model);
		double[] state = model.initialState();
		assertEquals(model.logDensity(state), cache.evaluate(model, state), 0.0);
		state[0] = 1.25;
		assertEquals(model.logDensity(state), cache.evaluate(model, state, 0), 1e-15);
		state[1] = -0.75;
		assertEquals(model.logDensity(state), cache.evaluate(model, state, 1), 1e-15);
	}

	@Test(expected = IllegalArgumentException.class)
	public void nutsDoesNotSilentlyFiniteDifference() {
		new NoUTurnSampler().sample(x -> -0.5 * x[0] * x[0], new double[] {0},
				SamplingOptions.builder().warmupIterations(0).sampleIterations(10).build(),
				new MersenneTwister(1));
	}

	@Test public void funnelRunsWithAnalyticGradientAndMultimodalityIsFlagged() {
		DifferentiableLogDensity funnel = new DifferentiableLogDensity() {
			@Override public double logDensityAndGradient(double[] state, double[] gradient) {
				double y = state[0], x = state[1];
				double inverseVariance = Math.exp(-y);
				gradient[0] = -y / 9.0 - 0.5 + 0.5 * x * x * inverseVariance;
				gradient[1] = -x * inverseVariance;
				return -0.5 * y * y / 9.0 - 0.5 * y
						- 0.5 * x * x * inverseVariance;
			}
		};
		assertTrue(Gradients.check(funnel, new double[] {0.3, -0.7}, 2e-6, 2e-6).passed());
		ChainResult funnelChain = new NoUTurnSampler().sample(funnel,
				new double[] {0, 0}, SamplingOptions.builder().warmupIterations(150)
						.sampleIterations(150).targetAcceptance(0.9).build(),
				new MersenneTwister(234));
		assertEquals(ChainResult.Status.SUCCESS, funnelChain.status());

		ChainResult[] modes = {
				syntheticChain(-5.0), syntheticChain(-5.1),
				syntheticChain(5.0), syntheticChain(5.1)};
		McmcDiagnosticReport report = McmcDiagnostics.analyze(new String[] {"mode"}, modes);
		assertTrue(report.parameter("mode").rHat() > 1.01);
		assertFalse(report.reliable());
	}

	private static ChainResult syntheticChain(double center) {
		double[][] samples = new double[100][1];
		double[] densities = new double[100];
		IterationStats[] stats = new IterationStats[100];
		for (int i = 0; i < samples.length; i++) {
			samples[i][0] = center + 0.01 * Math.sin(i);
			densities[i] = -0.5 * samples[i][0] * samples[i][0];
			stats[i] = new IterationStats(true, 0.8, 0.1, -densities[i], 0,
					false, 2, false, 3);
		}
		return new ChainResult(samples, densities, stats,
				new WarmupResult(0, 0.1, 0.1, new double[] {1}, Double.NaN),
				null, ChainResult.Status.SUCCESS, java.util.Collections.<String>emptyList());
	}
}
