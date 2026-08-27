/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

public class ModelScriptLanguageTest {
	@Test public void modernStanArraysMatricesAndGeneralBoundsExecute() {
		String source = "data { int<lower=1> N; array[N] real y; matrix[2,2] design; } "
				+ "parameters { vector<lower=-2,upper=3>[2] beta; real<upper=5> cap; "
				+ "real<lower=-3> floor_value; real<offset=10,multiplier=2> shifted; } "
				+ "model { y ~ normal(beta[1] + design[2,1]*beta[2],1); "
				+ "cap ~ normal(0,3); floor_value ~ normal(0,3); shifted ~ normal(10,3); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {3});
		data.put("y", new double[] {-0.2, 0.1, 0.4});
		data.put("design", new double[] {1, 2, 3, 4});
		BayesianModel model = ModelScript.compile(source, data).model();
		assertEquals(5, model.initialState().length);
		assertTrue(Double.isFinite(model.logDensity(model.initialState())));
		GradientCheckResult check = Gradients.check(model, model.initialState(), 3e-5, 3e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void distributionArgumentsBroadcastSymmetrically() {
		String source = "data { array[3] real y; } parameters { vector[3] mu; } "
				+ "model { mu ~ normal(0,2); y ~ normal(mu,1); "
				+ "target += normal_lpdf(y | mu,1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("y", new double[] {-0.4, 0.2, 0.7});
		BayesianModel model = ModelScript.compile(source, data).model();
		GradientCheckResult check = Gradients.check(model,
				new double[] {-0.1, 0.1, 0.3}, 3e-6, 3e-6);
		assertTrue(check.message(), check.passed());
	}

	@Test public void scalarUserFunctionsBranchesAndOverloadsExecute() {
		String source = "functions { "
				+ "real centered(real x, real mean) { real delta = x-mean; return delta; } "
				+ "real penalty(real x) { if (x > 0) return square(x); else return square(x)/2; } "
				+ "real magnitude(real x) { return x >= 0 ? x : -x; } "
				+ "real select(real x) { return x; } int select(int x) { return x+10; } "
				+ "} data { real y; } parameters { real mu; } model { "
				+ "target += normal_lpdf(centered(y,mu) | 0,1); "
				+ "target += -0.1*penalty(mu) + 0*magnitude(mu) + 0*select(1) + 0*select(mu); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("y", new double[] {0.4});
		BayesianModel model = ModelScript.compile(source, data).model();
		GradientCheckResult positive = Gradients.check(model, new double[] {0.2}, 3e-6, 3e-6);
		GradientCheckResult negative = Gradients.check(model, new double[] {-0.2}, 3e-6, 3e-6);
		assertTrue(positive.message(), positive.passed());
		assertTrue(negative.message(), negative.passed());
	}

	@Test public void controlFlowLocalsAndStanProbabilitySeparatorExecute() {
		String source = "data { int N; vector[N] y; } parameters { real mu; } "
				+ "model { real total = 0; for (n in 1:N) { "
				+ "if (y[n] >= 0) total += normal_lpdf(y[n] | mu, 1); "
				+ "else total += normal_lpdf(y[n] | mu, 2); } "
				+ "int pass = 0; while (pass < 2) { total += 0; pass += 1; } "
				+ "target += total; }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {4});
		data.put("y", new double[] {-0.4, 0.1, 0.3, -0.2});
		BayesianModel model = ModelScript.compile(source, data).model();
		assertEquals(1, model.factors().size());
		double expected = Normal.density(-0.4, 0.0, 2.0, true)
				+ Normal.density(0.1, 0.0, 1.0, true)
				+ Normal.density(0.3, 0.0, 1.0, true)
				+ Normal.density(-0.2, 0.0, 2.0, true);
		assertEquals(expected, model.logDensity(new double[] {0.0}), 2e-14);
		GradientCheckResult check = Gradients.check(model, new double[] {0.25}, 3e-6, 3e-6);
		assertTrue(check.message(), check.passed());
	}

	@Test public void simpleScriptsRetainDependencyAwareFactors() {
		BayesianModel model = ModelScript.compile("parameters { real x; } model { "
				+ "x ~ normal(0, 1); target += normal_lpdf(x | 1, 2); }").model();
		assertEquals(2, model.factors().size());
	}

	@Test public void scalarMathCatalogRetainsAnalyticGradients() {
		String source = "parameters { real x; } model { target += "
				+ "sin(x) + cos(x) + tanh(x) + asinh(x) + expm1(x) "
				+ "+ log1p_exp(x) + log_inv_logit(x) + log1m_inv_logit(x) "
				+ "+ erf(x) + lgamma(x + 2) + digamma(x + 2) "
				+ "+ atan2(x + 1, x + 2) + hypot(x + 1, x + 2) "
				+ "+ sinpi(x) + cospi(x) + tanpi(x / 4) + pow(x + 2, 2) + fma(x, 2, 1) "
				+ "+ log_sum_exp(x, 1) + log_mix(inv_logit(x), -1, -2) "
				+ "+ log_inv_logit_diff(x + 1, x) + multiply_log(x + 2, x + 3) "
				+ "+ log_rising_factorial(x + 3, 2) + log_falling_factorial(x + 3, 2) "
				+ "+ lbeta(x + 2, 3) + pi() * 0; }";
		BayesianModel model = ModelScript.compile(source).model();
		GradientCheckResult check = Gradients.check(model, new double[] {0.2}, 2e-5, 2e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void expandedStanScalarDistributionsCompileAndDifferentiate() {
		String source = "parameters { real mu; real<lower=0> scale; "
				+ "real<lower=0> shape; real<lower=0,upper=1> p; } model { "
				+ "mu ~ logistic(0, 2); scale ~ lognormal(0, 1); shape ~ gamma(2, 1); p ~ beta(2, 3); "
				+ "target += student_t_lpdf(0.3 | shape + 1, mu, scale); "
				+ "target += double_exponential_lpdf(-0.2 | mu, scale); "
				+ "target += gumbel_lpdf(0.5 | mu, scale); "
				+ "target += skew_normal_lpdf(0.2 | mu, scale, shape); "
				+ "target += exp_mod_normal_lpdf(0.2 | mu, scale, shape); "
				+ "target += von_mises_lpdf(0.2 | mu, shape); "
				+ "target += inv_gamma_lpdf(1.2 | shape, scale); "
				+ "target += chi_square_lpdf(1.3 | shape + 1); "
				+ "target += scaled_inv_chi_square_lpdf(1.4 | shape + 1, scale); "
				+ "target += weibull_lpdf(1.2 | shape, scale); "
				+ "target += frechet_lpdf(1.5 | shape, scale); "
				+ "target += rayleigh_lpdf(1.1 | scale); "
				+ "target += beta_proportion_lpdf(0.4 | p, shape + 1); "
				+ "target += pareto_lpdf(2.0 | 1, shape); "
				+ "target += pareto_type_2_lpdf(2.0 | 0, scale, shape); "
				+ "1 ~ bernoulli_logit(mu); 2 ~ binomial_logit(5, mu); "
				+ "2 ~ beta_binomial(5, shape, scale); 2 ~ neg_binomial(shape, scale); "
				+ "2 ~ neg_binomial_2(scale, shape); 2 ~ neg_binomial_2_log(mu, shape); "
				+ "2 ~ poisson_log(mu); 2 ~ geometric(p); 2 ~ discrete_range(0, 5); }";
		BayesianModel model = ModelScript.compile(source).model();
		assertTrue(Double.isFinite(model.logDensity(model.initialState())));
		GradientCheckResult check = Gradients.check(model, model.initialState(), 5e-5, 5e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void expandedRandomGeneratorsProduceSupportedValues() {
		String source = "parameters { real dummy; } model { dummy ~ normal(0, 1); } "
				+ "generated quantities { "
				+ "real a = student_t_rng(5, 0, 1); real b = weibull_rng(2, 1); "
				+ "real c = beta_rng(2, 3); int d = neg_binomial_2_rng(4, 2); "
				+ "int e = poisson_log_rng(0); int f = discrete_range_rng(-2, 2); "
				+ "real g = exp_mod_normal_rng(0, 1, 2); real h = von_mises_rng(0, 2); }";
		CompiledModelScript compiled = ModelScript.compile(source);
		Map<String, double[]> generated = compiled.generate(new double[] {0}, new MersenneTwister(81));
		assertTrue(Double.isFinite(generated.get("a")[0]));
		assertTrue(generated.get("b")[0] > 0.0);
		assertTrue(generated.get("c")[0] >= 0.0 && generated.get("c")[0] <= 1.0);
		assertTrue(generated.get("d")[0] >= 0.0);
		assertTrue(generated.get("e")[0] >= 0.0);
		assertTrue(generated.get("f")[0] >= -2.0 && generated.get("f")[0] <= 2.0);
		assertTrue(Double.isFinite(generated.get("g")[0]));
		assertTrue(generated.get("h")[0] >= -Math.PI && generated.get("h")[0] <= Math.PI);
	}

	@Test public void scalarDistributionParameterizationsMatchJdistlib() {
		assertEquals(LogNormal.density(1.4, 0.2, 0.8, true),
				scriptContribution("lognormal_lpdf(1.4 | 0.2, 0.8)"), 2e-14);
		assertEquals(T.density((0.4 - 0.1) / 1.3, 6.0, true) - Math.log(1.3),
				scriptContribution("student_t_lpdf(0.4 | 6, 0.1, 1.3)"), 2e-14);
		assertEquals(Laplace.density(-0.3, 0.2, 1.1, true),
				scriptContribution("double_exponential_lpdf(-0.3 | 0.2, 1.1)"), 2e-14);
		assertEquals(Logistic.density(0.7, -0.1, 1.2, true),
				scriptContribution("logistic_lpdf(0.7 | -0.1, 1.2)"), 2e-14);
		assertEquals(ExponentiallyModifiedGaussian.density(0.7, -0.1, 1.2, 0.9, true),
				scriptContribution("exp_mod_normal_lpdf(0.7 | -0.1, 1.2, 0.9)"), 2e-14);
		assertEquals(Gamma.density(1.3, 2.2, 1.0 / 0.7, true),
				scriptContribution("gamma_lpdf(1.3 | 2.2, 0.7)"), 2e-14);
		assertEquals(Weibull.density(1.3, 2.2, 0.7, true),
				scriptContribution("weibull_lpdf(1.3 | 2.2, 0.7)"), 2e-14);
		assertEquals(NegBinomial.density(3, 2.2, 0.7 / 1.7, true),
				scriptContribution("neg_binomial_lpmf(3 | 2.2, 0.7)"), 2e-14);
	}

	private static double scriptContribution(String expression) {
		String source = "parameters { real dummy; } model { dummy ~ normal(0, 1); "
				+ "target += " + expression + "; }";
		double total = ModelScript.compile(source).model().logDensity(new double[] {0});
		return total - Normal.density(0, 0, 1, true);
	}
}
