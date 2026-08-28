/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

public class AdvancedProbabilityFeaturesTest {
	private static final MultivariateProbabilityOptions ACCURATE =
			new MultivariateProbabilityOptions(3e-6, 3e-6, 262144, 12);

	@Test
	public void countRectangleDynamicProgramsMatchBruteForceEnumeration() {
		int size = 7;
		int[] lower = {1, 0, 2};
		int[] upper = {4, 3, 5};
		double[] weights = {0.2, 0.3, 0.5};
		double[] alpha = {0.7, 1.4, 2.1};
		int[] population = {5, 7, 9};
		double multinomial = 0.0;
		double dirichletMultinomial = 0.0;
		double hypergeometric = 0.0;
		for (int first = 0; first <= size; first++) {
			for (int second = 0; second <= size - first; second++) {
				int third = size - first - second;
				if (first < lower[0] || first > upper[0] || second < lower[1] ||
						second > upper[1] || third < lower[2] || third > upper[2]) continue;
				double[] counts = {first, second, third};
				multinomial += Multinomial.density(counts, size, weights, false);
				dirichletMultinomial += DirichletMultinomial.density(counts, size,
						alpha, false);
				hypergeometric += MultivariateHypergeometric.density(
						new int[] {first, second, third}, population, size, false);
			}
		}
		assertEquals(multinomial, Multinomial.probability(lower, upper, size,
				weights), 2e-15);
		assertEquals(dirichletMultinomial, DirichletMultinomial.probability(lower,
				upper, size, alpha), 3e-15);
		assertEquals(hypergeometric, MultivariateHypergeometric.probability(lower,
				upper, population, size), 1e-14);
		assertEquals(Multinomial.probability(new int[3], upper, size, weights),
				Multinomial.cumulative(upper, size, weights), 0.0);
	}

	@Test
	public void dirichletProbabilitiesRespectSimplexGeometry() {
		MultivariateProbabilityResult beta = Dirichlet.probability(
				new double[] {0.2, 0.1}, new double[] {0.7, 0.8},
				new double[] {2.0, 3.0});
		double expected = Beta.cumulative(0.7, 2.0, 3.0, true, false) -
				Beta.cumulative(0.2, 2.0, 3.0, true, false);
		assertEquals(expected, beta.probability, 2e-15);
		assertEquals(0, beta.evaluations);

		MultivariateProbabilityResult uniformSimplex = Dirichlet.cumulative(
				new double[] {0.5, 0.5, 0.5}, new double[] {1.0, 1.0, 1.0},
				ACCURATE, new MersenneTwister(31L));
		assertEquals(0.25, uniformSimplex.probability, 2e-5);
		assertTrue(uniformSimplex.absoluteError <= 3e-6);
	}

	@Test
	public void ellipticalOrthantsUseMixtureAndRadialConditioning() {
		double[] location = {0.0, 0.0};
		double[][] scatter = {{1.0, 0.5}, {0.5, 1.0}};
		double expected = 1.0 / 3.0;
		MultivariateProbabilityResult laplace = MultivariateLaplace.cumulative(
				new double[] {0.0, 0.0}, location, scatter, ACCURATE,
				new MersenneTwister(32L));
		assertEquals(expected, laplace.probability, 3e-5);
		MultivariateProbabilityResult heavy =
				MultivariatePowerExponential.cumulative(new double[] {0.0, 0.0},
						location, scatter, 0.55, ACCURATE, new MersenneTwister(33L));
		assertEquals(expected, heavy.probability, 5e-5);
		MultivariateProbabilityResult light =
				MultivariatePowerExponential.cumulative(new double[] {0.0, 0.0},
						location, scatter, 1.8, ACCURATE, new MersenneTwister(34L));
		assertEquals(expected, light.probability, 5e-5);
	}

	@Test
	public void powerExponentialGaussianCaseDelegatesExactly() {
		double[] lower = {-1.0, -0.2};
		double[] upper = {0.7, 1.3};
		double[] location = {0.1, -0.4};
		double[][] scatter = {{1.2, 0.3}, {0.3, 0.9}};
		MultivariateProbabilityResult expected = MultivariateNormal.probability(
				lower, upper, location, scatter, ACCURATE, new MersenneTwister(35L));
		MultivariateProbabilityResult actual =
				MultivariatePowerExponential.probability(lower, upper, location,
						scatter, 1.0, ACCURATE, new MersenneTwister(35L));
		assertEquals(expected.probability, actual.probability, 0.0);
		assertEquals(expected.absoluteError, actual.absoluteError, 0.0);
	}

	@Test
	public void wienerDensityCdfRngAndScriptCatalogAgree() {
		double boundary = 1.3;
		double nondecision = 0.18;
		double bias = 0.37;
		double drift = 0.8;
		assertTrue(Wiener.density(0.181, boundary, nondecision, bias, drift,
				false) > 0.0);
		assertTrue(Wiener.density(8.0, boundary, nondecision, bias, drift,
				false) > 0.0);
		double mass = Wiener.boundaryProbability(boundary, bias, drift);
		assertEquals(mass, Wiener.cumulative(40.0, boundary, nondecision, bias,
				drift), 2e-8);
		double draw = Wiener.random(boundary, nondecision, bias, drift,
				new MersenneTwister(36L));
		assertTrue(draw >= nondecision && Double.isFinite(draw));

		String source = "data { real y; } parameters { real delta; } " +
				"model { y ~ wiener(1.3, 0.18, 0.37, delta); }";
		BayesianModel model = ModelScript.compileStan(source,
				Collections.singletonMap("y", new double[] {0.9})).model();
		assertEquals(Wiener.density(0.9, boundary, nondecision, bias, drift, true),
				model.logDensity(new double[] {drift}), 1e-12);
	}

	@Test
	public void wienerLogDensityStaysDefinedAcrossExtremeRegimes() {
		double[] boundaries = {0.1, 1.0, 10.0};
		double[] biases = {1e-4, 0.5, 1.0 - 1e-4};
		double[] drifts = {-20.0, 0.0, 20.0};
		double[] scaledTimes = {1e-10, 1e-5, 0.249, 0.251, 10.0};
		for (double a : boundaries) for (double beta : biases)
			for (double delta : drifts) for (double scaledTime : scaledTimes) {
				double logDensity = Wiener.density(a * a * scaledTime, a, 0.0,
						beta, delta, true);
				assertTrue("a=" + a + ", beta=" + beta + ", drift=" + delta +
						", t*=" + scaledTime, Double.isFinite(logDensity));
			}
	}
}
