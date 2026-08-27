/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Constraints;
import jdistlib.inference.DifferentiableLogDensity;
import jdistlib.inference.GibbsSampler;
import jdistlib.inference.Gradients;
import jdistlib.inference.MetropolisBlockKernel;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.RandomWalkMetropolis;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

/** Fifteen executable, canonical Bayesian examples used by the inference guide. */
public class BayesianShowcaseTest {
	@Test public void example01BetaBinomialCoinToss() {
		CompiledModelScript script = compile(
				"data { int n; int y; } parameters { real<lower=0, upper=1> theta; } "
				+ "model { theta ~ beta(2,2); y ~ binomial(n,theta); } "
				+ "generated quantities { int y_rep = binomial_rng(n,theta); }",
				"n", 10, "y", 7);
		assertGradient(script.model());
		assertTrue(Double.isFinite(script.model().logDensity(script.model().initialState())));
	}

	@Test public void example02NormalMeanKnownVariance() {
		BayesianModel model = new ModelBuilder().data("y", 0.8, 1.1, 1.4, 0.9)
				.parameter("mu", Constraints.real(), 0.0)
				.factor("prior", new String[] {"mu"}, ModelFactors.normalPrior("mu", 0, 5))
				.factor("likelihood", new String[] {"y", "mu"},
						ModelFactors.normalObservations("y", "mu", 1)).build();
		assertGradient(model);
		assertSuccessful(new NoUTurnSampler().sample(model, model.initialState(), shortNuts(),
				new MersenneTwister(102)));
	}

	@Test public void example03NormalLocationAndScale() {
		CompiledModelScript script = compileVector(
				"data { int N; vector[N] y; } parameters { real mu; real<lower=0> sigma; } "
				+ "model { mu ~ normal(0,10); sigma ~ exponential(1); y ~ normal(mu,sigma); } "
				+ "generated quantities { real y_rep = normal_rng(mu,sigma); }",
				new double[] {-0.2, 0.1, 0.4, 0.0});
		assertGradient(script.model());
	}

	@Test public void example04GammaPoissonCountRate() {
		CompiledModelScript script = compile(
				"data { int y; } parameters { real<lower=0> lambda; } "
				+ "model { lambda ~ gamma(2,1); y ~ poisson(lambda); } "
				+ "generated quantities { int y_rep = poisson_rng(lambda); }", "y", 12);
		assertGradient(script.model());
		assertTrue(script.generate(script.model().initialState(), new MersenneTwister(4))
				.get("y_rep")[0] >= 0.0);
	}

	@Test public void example05BayesianLinearRegression() {
		final double[] predictor = {-1, 0, 1, 2};
		final double[] response = {-0.8, 1.1, 2.9, 5.2};
		DifferentiableLogDensity target = (state, gradient) -> {
			double intercept = state[0], slope = state[1];
			double result = -0.5 * (intercept * intercept + slope * slope) / 25.0;
			gradient[0] = -intercept / 25.0; gradient[1] = -slope / 25.0;
			for (int i = 0; i < response.length; i++) {
				double residual = response[i] - intercept - slope * predictor[i];
				result -= 0.5 * residual * residual;
				gradient[0] += residual; gradient[1] += residual * predictor[i];
			}
			return result;
		};
		assertGradient(target, new double[] {0.5, 1.5});
	}

	@Test public void example06BayesianLogisticRegression() {
		final double[] predictor = {-2, -1, 1, 2};
		final int[] outcome = {0, 0, 1, 1};
		DifferentiableLogDensity target = (state, gradient) -> {
			double result = -0.5 * (state[0] * state[0] + state[1] * state[1]) / 25.0;
			gradient[0] = -state[0] / 25.0; gradient[1] = -state[1] / 25.0;
			for (int i = 0; i < outcome.length; i++) {
				double eta = state[0] + state[1] * predictor[i];
				double residual = outcome[i] - logistic(eta);
				result += outcome[i] * eta - log1pExp(eta);
				gradient[0] += residual; gradient[1] += residual * predictor[i];
			}
			return result;
		};
		assertGradient(target, new double[] {-0.1, 0.7});
	}

	@Test public void example07EightSchoolsNonCentered() {
		final double[] y = {28, 8, -3, 7, -1, 1, 18, 12};
		final double[] sigma = {15, 10, 16, 11, 9, 11, 10, 18};
		DifferentiableLogDensity target = (state, gradient) -> {
			double mu = state[0], tau = Math.exp(state[1]);
			double result = -0.5 * mu * mu / 25.0 - 0.5 * tau * tau / 25.0
					+ state[1];
			gradient[0] = -mu / 25.0;
			gradient[1] = 1.0 - tau * tau / 25.0;
			for (int i = 0; i < 8; i++) {
				double z = state[i + 2], theta = mu + tau * z;
				double residual = y[i] - theta, inverseVariance = 1.0 / (sigma[i] * sigma[i]);
				result -= 0.5 * z * z + 0.5 * residual * residual * inverseVariance;
				gradient[0] += residual * inverseVariance;
				gradient[1] += residual * tau * z * inverseVariance;
				gradient[i + 2] = -z + residual * tau * inverseVariance;
			}
			return result;
		};
		assertGradient(target, new double[] {5, Math.log(3), 0, 0, 0, 0, 0, 0, 0, 0});
	}

	@Test public void example08NealsFunnel() {
		DifferentiableLogDensity target = (state, gradient) -> {
			double y = state[0], x = state[1], inverseVariance = Math.exp(-y);
			gradient[0] = -y / 9.0 - 0.5 + 0.5 * x * x * inverseVariance;
			gradient[1] = -x * inverseVariance;
			return -0.5 * y * y / 9.0 - 0.5 * y - 0.5 * x * x * inverseVariance;
		};
		assertGradient(target, new double[] {0.3, -0.7});
		assertSuccessful(new NoUTurnSampler().sample(target, new double[] {0, 0}, shortNuts(),
				new MersenneTwister(108)));
	}

	@Test public void example09RosenbrockBananaDensity() {
		DifferentiableLogDensity target = (state, gradient) -> {
			double first = 1.0 - state[0], second = state[1] - state[0] * state[0];
			gradient[0] = first + 100.0 * state[0] * second;
			gradient[1] = -50.0 * second;
			return -0.5 * first * first - 25.0 * second * second;
		};
		assertGradient(target, new double[] {-0.5, 0.7});
	}

	@Test public void example10TwoComponentGaussianMixture() {
		jdistlib.inference.LogDensity target = state -> {
			double left = -0.5 * (state[0] + 3) * (state[0] + 3);
			double right = -0.5 * (state[0] - 3) * (state[0] - 3);
			double maximum = Math.max(left, right);
			return maximum + Math.log(0.5 * Math.exp(left - maximum)
					+ 0.5 * Math.exp(right - maximum));
		};
		ChainResult chain = new RandomWalkMetropolis().sample(target, new double[] {-3},
				SamplingOptions.builder().warmupIterations(20).sampleIterations(40)
						.stepSize(1).build(), new MersenneTwister(110));
		assertSuccessful(chain);
	}

	@Test public void example11RobustCauchyLocation() {
		CompiledModelScript script = compileVector(
				"data { int N; vector[N] y; } parameters { real location; } "
				+ "model { location ~ normal(0,10); y ~ cauchy(location,1); }",
				new double[] {-0.1, 0.2, 0.0, 15.0});
		assertGradient(script.model());
	}

	@Test public void example12DirichletMultinomialSimplex() {
		final double[] counts = {12, 7, 3};
		BayesianModel model = new ModelBuilder()
				.parameter("probability", Constraints.simplex(3), 0.4, 0.35, 0.25)
				.factor("Dirichlet-multinomial", new String[] {"probability"}, state -> {
					double result = 0.0;
					for (int i = 0; i < counts.length; i++)
						result += (counts[i] + 1.0) * Math.log(state.value("probability", i));
					return result;
				}).build();
		assertTrue(Gradients.check(model, model.initialState(), 2e-5, 2e-5).passed());
		assertEquals(1.0, sum(model.state(model.initialState()).vector("probability")), 1e-14);
	}

	@Test public void example13PoissonChangePointGibbs() {
		final int[] counts = {2, 1, 3, 10, 12, 9};
		jdistlib.inference.LogDensity target = state -> {
			int split = (int) state[0];
			if (split != state[0] || split < 1 || split >= counts.length) return Double.NEGATIVE_INFINITY;
			double before = Math.exp(state[1]), after = Math.exp(state[2]);
			double result = state[1] - before + state[2] - after;
			for (int i = 0; i < counts.length; i++) {
				double rate = i < split ? before : after;
				result += counts[i] * Math.log(rate) - rate;
			}
			return result;
		};
		GibbsSampler sampler = new GibbsSampler((state, density, random) -> {
			double[] weights = new double[counts.length - 1]; double maximum = Double.NEGATIVE_INFINITY;
			for (int split = 1; split < counts.length; split++) {
				state[0] = split; weights[split - 1] = density.logDensity(state);
				maximum = Math.max(maximum, weights[split - 1]);
			}
			double total = 0.0;
			for (int i = 0; i < weights.length; i++) total += weights[i] = Math.exp(weights[i] - maximum);
			double draw = random.nextDouble() * total;
			for (int i = 0; i < weights.length; i++) if ((draw -= weights[i]) <= 0) { state[0] = i + 1; return; }
			state[0] = weights.length;
		}, new MetropolisBlockKernel(new int[] {1, 2}, 0.25, false));
		assertSuccessful(sampler.sample(target, new double[] {3, Math.log(2), Math.log(10)},
				SamplingOptions.builder().warmupIterations(20).sampleIterations(40).build(),
				new MersenneTwister(113)));
	}

	@Test public void example14CorrelatedGaussianDenseMetric() {
		DifferentiableLogDensity target = (state, gradient) -> {
			double rho = 0.9, denominator = 1.0 - rho * rho;
			gradient[0] = -(state[0] - rho * state[1]) / denominator;
			gradient[1] = -(state[1] - rho * state[0]) / denominator;
			return -(state[0] * state[0] - 2 * rho * state[0] * state[1]
					+ state[1] * state[1]) / (2 * denominator);
		};
		assertGradient(target, new double[] {0.2, -0.4});
		ChainResult chain = new NoUTurnSampler().sample(target, new double[] {0, 0},
				SamplingOptions.builder().warmupIterations(80).sampleIterations(20)
						.denseMassMatrix(true).build(), new MersenneTwister(114));
		assertSuccessful(chain);
		assertEquals(2, chain.warmup().inverseMassMatrix().length);
	}

	@Test public void example15BayesianAbTest() {
		CompiledModelScript script = compile(
				"data { int n_a; int y_a; int n_b; int y_b; } "
				+ "parameters { real<lower=0,upper=1> p_a; real<lower=0,upper=1> p_b; } "
				+ "model { p_a ~ beta(1,1); p_b ~ beta(1,1); "
				+ "y_a ~ binomial(n_a,p_a); y_b ~ binomial(n_b,p_b); }",
				"n_a", 100, "y_a", 12, "n_b", 100, "y_b", 18);
		assertGradient(script.model());
		assertSuccessful(new NoUTurnSampler().sample(script.model(), script.model().initialState(),
				shortNuts(), new MersenneTwister(115)));
	}

	private static SamplingOptions shortNuts() {
		return SamplingOptions.builder().warmupIterations(40).sampleIterations(30)
				.maximumTreeDepth(6).build();
	}
	private static void assertSuccessful(ChainResult result) {
		assertEquals(ChainResult.Status.SUCCESS, result.status());
		assertTrue(result.size() > 0);
	}
	private static void assertGradient(BayesianModel model) {
		assertTrue(Gradients.check(model, model.initialState(), 1e-5, 1e-5).passed());
	}
	private static void assertGradient(DifferentiableLogDensity target, double[] state) {
		assertTrue(Gradients.check(target, state, 1e-5, 1e-5).passed());
	}
	private static CompiledModelScript compile(String source, Object... pairs) {
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		for (int i = 0; i < pairs.length; i += 2)
			data.put((String) pairs[i], new double[] {((Number) pairs[i + 1]).doubleValue()});
		return ModelScript.compile(source, data);
	}
	private static CompiledModelScript compileVector(String source, double[] values) {
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {values.length}); data.put("y", values);
		return ModelScript.compile(source, data);
	}
	private static double logistic(double value) {
		return value >= 0 ? 1.0 / (1.0 + Math.exp(-value))
				: Math.exp(value) / (1.0 + Math.exp(value));
	}
	private static double log1pExp(double value) {
		return value > 0 ? value + Math.log1p(Math.exp(-value)) : Math.log1p(Math.exp(value));
	}
	private static double sum(double[] values) {
		double result = 0.0; for (double value : values) result += value; return result;
	}
}
