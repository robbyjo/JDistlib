/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class CopulaTest {
	@Test
	public void independenceHasProductCdfAndUnitDensity() {
		Copula copula = new IndependenceCopula(3);
		double[] u = {0.2, 0.5, 0.8};
		assertEquals(0.08, copula.cumulative(u), 1e-16);
		assertEquals(1.0, copula.density(u), 0.0);
		assertEquals(0.0, copula.kendallsTau(0, 2), 0.0);
		assertEquals(1.0, copula.kendallsTau(1, 1), 0.0);
		assertTrue(copula.diagnose(new double[] {0.0, 0.5, 1.0})
				.isDensityDefined());
	}

	@Test
	public void everyFamilyHasUniformMarginsAndExpectedExampleValues() {
		double rho = 0.5;
		GaussianCopula gaussian = new GaussianCopula(
				new double[][] {{1.0, rho}, {rho, 1.0}});
		assertEquals(1.0 / Math.sqrt(1.0 - rho * rho),
				gaussian.density(new double[] {0.5, 0.5}), 2e-15);
		assertEquals(1.0 / 3.0, gaussian.cumulative(new double[] {0.5, 0.5}), 2e-4);

		ClaytonCopula clayton = new ClaytonCopula(2, 2.0);
		double[] point = {0.4, 0.7};
		double sum = Math.pow(point[0], -2.0) + Math.pow(point[1], -2.0) - 1.0;
		assertEquals(Math.pow(sum, -0.5), clayton.cumulative(point), 2e-16);
		double claytonDensity = 3.0 * Math.pow(point[0] * point[1], -3.0)
				* Math.pow(sum, -2.5);
		assertEquals(claytonDensity, clayton.density(point), 2e-15);

		FrankCopula frank = new FrankCopula(2, 3.0);
		double numerator = Math.expm1(-3.0 * point[0]) * Math.expm1(-3.0 * point[1]);
		assertEquals(-Math.log1p(numerator / Math.expm1(-3.0)) / 3.0,
				frank.cumulative(point), 2e-16);

		Copula[] copulas = {gaussian, new StudentTCopula(
				new double[][] {{1.0, rho}, {rho, 1.0}}, 5.0), clayton,
				new GumbelCopula(2, 1.7), frank, new FrankCopula(2, -3.0)};
		for (Copula copula : copulas) {
			assertEquals(0.37, copula.cumulative(new double[] {0.37, 1.0}), 3e-4);
			assertEquals(0.62, copula.cumulative(new double[] {1.0, 0.62}), 3e-4);
			assertEquals(0.0, copula.cumulative(new double[] {0.0, 0.8}), 0.0);
			assertEquals(1.0, copula.cumulative(new double[] {1.0, 1.0}), 0.0);
		}
	}

	@Test
	public void archimedeanDensitiesNumericallyNormalize() {
		Copula[] copulas = {new ClaytonCopula(2, 1.4), new GumbelCopula(2, 1.8),
				new FrankCopula(2, 4.0), new FrankCopula(2, -4.0)};
		int divisions = 240;
		for (Copula copula : copulas) {
			double sum = 0.0;
			for (int i = 0; i < divisions; i++) {
				for (int j = 0; j < divisions; j++) {
					sum += copula.density(new double[] {
							(i + 0.5) / divisions, (j + 0.5) / divisions});
				}
			}
			assertEquals(copula.getClass().getSimpleName(), 1.0,
					sum / (divisions * divisions), 0.012);
		}
	}

	@Test
	public void higherDimensionalArchimedeanDerivativesNormalize() {
		Copula[] copulas = {new ClaytonCopula(3, 0.7), new GumbelCopula(3, 1.4),
				new FrankCopula(3, 2.0)};
		int divisions = 34;
		for (Copula copula : copulas) {
			double sum = 0.0;
			for (int i = 0; i < divisions; i++) {
				for (int j = 0; j < divisions; j++) {
					for (int k = 0; k < divisions; k++) {
						sum += copula.density(new double[] {(i + 0.5) / divisions,
								(j + 0.5) / divisions, (k + 0.5) / divisions});
					}
				}
			}
			assertEquals(copula.getClass().getSimpleName(), 1.0,
					sum / (divisions * divisions * divisions), 0.025);
		}
	}

	@Test
	public void parameterConversionsRoundTripKendallsTau() {
		assertEquals(0.63, new ClaytonCopula(3,
				ClaytonCopula.parameterFromKendallsTau(0.63)).kendallsTau(0, 1), 1e-15);
		assertEquals(0.63, new GumbelCopula(3,
				GumbelCopula.parameterFromKendallsTau(0.63)).kendallsTau(0, 1), 1e-15);
		for (double tau : new double[] {-0.7, -0.2, 0.0, 0.2, 0.7}) {
			FrankCopula copula = FrankCopula.fromKendallsTau(2, tau);
			assertEquals(tau, copula.kendallsTau(0, 1), 2e-11);
		}

		double[][] tau = {{1.0, -0.25}, {-0.25, 1.0}};
		GaussianCopula gaussian = GaussianCopula.fromKendallsTau(tau);
		StudentTCopula student = StudentTCopula.fromKendallsTau(tau, 4.0);
		assertEquals(-0.25, gaussian.kendallsTau(0, 1), 2e-16);
		assertEquals(-0.25, student.kendallsTau(0, 1), 2e-16);
	}

	@Test
	public void samplersAreReproducibleAndMatchDependence() {
		Copula[] copulas = {
				new GaussianCopula(new double[][] {{1.0, 0.65}, {0.65, 1.0}}),
				new StudentTCopula(new double[][] {{1.0, -0.45}, {-0.45, 1.0}}, 4.0),
				new ClaytonCopula(2, 2.0), new GumbelCopula(2, 2.0),
				new FrankCopula(2, 5.0), new FrankCopula(2, -5.0)};
		for (Copula copula : copulas) {
			double[][] first = copula.random(900, 20260826L);
			double[][] second = copula.random(900, 20260826L);
			assertArrayEquals(first[0], second[0], 0.0);
			for (double[] row : first) {
				assertTrue(row[0] >= 0.0 && row[0] <= 1.0);
				assertTrue(row[1] >= 0.0 && row[1] <= 1.0);
			}
			assertEquals(copula.getClass().getSimpleName(), copula.kendallsTau(0, 1),
					empiricalTau(first), 0.065);
		}
	}

	@Test
	public void compositionAppliesMarginalJacobiansAndQuantiles() {
		CopulaDistribution distribution = new CopulaDistribution(
				new IndependenceCopula(2), new Normal(1.0, 2.0), new Exponential(3.0));
		double[] x = {0.4, 2.5};
		assertEquals(Normal.cumulative(x[0], 1.0, 2.0, true, false)
				* Exponential.cumulative(x[1], 3.0, true, false),
				distribution.cumulative(x), 2e-16);
		assertEquals(Normal.density(x[0], 1.0, 2.0, false)
				* Exponential.density(x[1], 3.0, false), distribution.density(x), 2e-16);
		assertArrayEquals(distribution.random(17L), distribution.random(17L), 0.0);
	}

	@Test
	public void validationAndBoundaryDiagnosticsAreExplicit() {
		Copula copula = new ClaytonCopula(2, 1.0);
		assertTrue(copula.diagnose(new double[] {0.2, 0.8}).isDensityDefined());
		CopulaDiagnostics boundary = copula.diagnose(new double[] {0.0, 1.0});
		assertTrue(boundary.isBoundary());
		assertFalse(boundary.isDensityDefined());
		assertEquals(1, boundary.getLowerBoundaryCoordinates());
		assertEquals(1, boundary.getUpperBoundaryCoordinates());
		assertTrue(Double.isNaN(copula.logDensity(new double[] {0.0, 1.0})));
		assertFalse(copula.diagnose(new double[] {-0.1, 0.5}).isValid());

		assertThrows(IllegalArgumentException.class,
				() -> new GaussianCopula(new double[][] {{1.0, 2.0}, {2.0, 1.0}}));
		assertThrows(IllegalArgumentException.class, () -> new ClaytonCopula(2, -0.1));
		assertThrows(IllegalArgumentException.class, () -> new GumbelCopula(2, 0.9));
		assertThrows(IllegalArgumentException.class, () -> new FrankCopula(3, -1.0));
		assertThrows(IllegalArgumentException.class,
				() -> new CopulaDistribution(new IndependenceCopula(2), new Normal()));
	}

	@Test
	public void strongFrankParametersRemainNumericallyUsable() {
		for (double theta : new double[] {100.0, -100.0}) {
			FrankCopula copula = new FrankCopula(2, theta);
			double[] point = {0.35, 0.68};
			assertTrue(Double.isFinite(copula.cumulative(point)));
			assertTrue(Double.isFinite(copula.logDensity(point)));
			double[] draw = copula.random(91L);
			assertTrue(draw[0] > 0.0 && draw[0] < 1.0);
			assertTrue(draw[1] > 0.0 && draw[1] < 1.0);
		}
	}

	@Test
	public void independenceLimitsRemainStable() {
		double[] point = {0.31, 0.74};
		double product = point[0] * point[1];
		Copula[] copulas = {new ClaytonCopula(2, 1e-9),
				new GumbelCopula(2, 1.0 + 1e-9), new FrankCopula(2, 1e-7)};
		for (Copula copula : copulas) {
			assertEquals(copula.getClass().getSimpleName(), product,
					copula.cumulative(point), 2e-8);
			assertEquals(copula.getClass().getSimpleName(), 1.0,
					copula.density(point), 2e-7);
		}
	}

	private static double empiricalTau(double[][] sample) {
		long concordant = 0;
		long discordant = 0;
		for (int i = 0; i < sample.length; i++) {
			for (int j = i + 1; j < sample.length; j++) {
				double product = (sample[i][0] - sample[j][0])
						* (sample[i][1] - sample[j][1]);
				if (product > 0.0) concordant++;
				else if (product < 0.0) discordant++;
			}
		}
		return (double) (concordant - discordant) / (concordant + discordant);
	}
}
