/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class MultivariateDistributionsTest {
	@Test
	public void simplexAndCountMassesMatchClosedForms() {
		assertEquals(2.0, Dirichlet.density(new double[] {0.2, 0.3, 0.5},
				new double[] {1.0, 1.0, 1.0}, false), 2e-15);
		assertEquals(7.56, Dirichlet.density(new double[] {0.2, 0.3, 0.5},
				new double[] {2.0, 3.0, 4.0}, false), 2e-14);
		assertEquals(1.0 / 6.0, DirichletMultinomial.density(
				new double[] {1.0, 1.0, 0.0}, 2,
				new double[] {1.0, 1.0, 1.0}, false), 2e-15);
		assertEquals(2.0 / 7.0, MultivariateHypergeometric.density(
				new int[] {2, 1, 1}, new int[] {5, 3, 2}, 4, false), 2e-15);
	}

	@Test
	public void bivariatePoissonUsesSharedComponentDefinition() {
		double expectedAtOrigin = Math.exp(-3.5);
		assertEquals(expectedAtOrigin,
				BivariatePoisson.density(0, 0, 1.0, 2.0, 0.5, false), 2e-16);
		assertEquals(expectedAtOrigin,
				BivariatePoisson.cumulative(0, 0, 1.0, 2.0, 0.5, false), 2e-16);
		assertEquals(Math.log(expectedAtOrigin),
				BivariatePoisson.density(0, 0, 1.0, 2.0, 0.5, true), 2e-15);
	}

	@Test
	public void vgamBivariateLogisticMatchesItsDefinition() {
		assertEquals(1.0 / 3.0, BivariateLogistic.cumulative(0.0, 0.0,
				0.0, 1.0, 0.0, 1.0, false), 2e-16);
		assertEquals(2.0 / 27.0, BivariateLogistic.density(0.0, 0.0,
				0.0, 1.0, 0.0, 1.0, false), 2e-16);
		assertEquals(0.0, BivariateLogistic.cumulative(Double.NEGATIVE_INFINITY,
				0.0, 0.0, 1.0, 0.0, 1.0, false), 0.0);
	}

	@Test
	public void ellipticalLawsReduceToUnivariateAndGaussianCases() {
		double[] location = {0.3};
		double[][] scale = {{2.25}};
		double x = 1.1;
		assertEquals(Normal.density(x, 0.3, 1.5, false),
				MultivariateNormal.density(new double[] {x}, location, scale, false),
				2e-16);
		assertEquals(T.density((x - 0.3) / 1.5, 7.0, false) / 1.5,
				MultivariateStudentT.density(new double[] {x}, location, scale, 7.0,
						false), 2e-16);
		assertEquals(Cauchy.density(x, 0.3, 1.5, false),
				MultivariateCauchy.density(new double[] {x}, location, scale, false),
				2e-16);
		assertEquals(MultivariateNormal.density(new double[] {x}, location, scale,
				false), MultivariatePowerExponential.density(new double[] {x}, location,
				scale, 1.0, false), 2e-16);
	}

	@Test
	public void transformedAndMixtureDensitiesHaveExpectedJacobians() {
		double[] mean = {0.1, -0.2};
		double[][] covariance = {{1.0, 0.25}, {0.25, 0.8}};
		double[] x = {Math.exp(0.4), Math.exp(-0.7)};
		double normalLog = MultivariateNormal.density(new double[] {0.4, -0.7},
				mean, covariance, true);
		assertEquals(normalLog - Math.log(x[0]) - Math.log(x[1]),
				MultivariateLogNormal.density(x, mean, covariance, true), 2e-15);
		assertEquals(Laplace.density(1.0, 0.0, 1.0 / Math.sqrt(2.0), false),
				MultivariateLaplace.density(new double[] {1.0}, new double[] {0.0},
						new double[][] {{1.0}}, false), 2e-15);
	}

	@Test
	public void invalidParametersAndSupportAreRejected() {
		assertTrue(Double.isNaN(MultivariateNormal.density(new double[] {0, 0},
				new double[] {0, 0}, new double[][] {{1, 2}, {2, 1}}, false)));
		assertNull(MultivariateStudentT.random(new double[] {0},
				new double[][] {{1}}, 0.0, new MersenneTwister(1L)));
		assertEquals(0.0, MultivariateLogNormal.density(new double[] {-1.0},
				new double[] {0.0}, new double[][] {{1.0}}, false), 0.0);
		assertEquals(0.0, Dirichlet.density(new double[] {0.2, 0.2},
				new double[] {1.0, 1.0}, false), 0.0);
	}

	@Test
	public void randomVectorsRespectSupportAndMoments() {
		MersenneTwister random = new MersenneTwister(20260824L);
		int n = 30000;
		double[] dirichletSum = new double[3];
		double[] normalSum = new double[2];
		double[] normalProductSum = new double[3];
		double[] poissonSum = new double[3];
		for (int i = 0; i < n; i++) {
			double[] d = Dirichlet.random(new double[] {2, 3, 5}, random);
			assertNotNull(d);
			assertEquals(1.0, d[0] + d[1] + d[2], 2e-15);
			for (int j = 0; j < d.length; j++) dirichletSum[j] += d[j];

			double[] z = MultivariateNormal.random(new double[] {1, -2},
					new double[][] {{2, 0.6}, {0.6, 1}}, random);
			normalSum[0] += z[0];
			normalSum[1] += z[1];
			normalProductSum[0] += (z[0] - 1) * (z[0] - 1);
			normalProductSum[1] += (z[1] + 2) * (z[1] + 2);
			normalProductSum[2] += (z[0] - 1) * (z[1] + 2);

			int[] p = BivariatePoisson.random(1.0, 2.0, 0.5, random);
			poissonSum[0] += p[0];
			poissonSum[1] += p[1];
			poissonSum[2] += p[0] * p[1];
		}
		assertArrayEquals(new double[] {0.2, 0.3, 0.5},
				new double[] {dirichletSum[0] / n, dirichletSum[1] / n,
					dirichletSum[2] / n}, 0.008);
		assertEquals(1.0, normalSum[0] / n, 0.025);
		assertEquals(-2.0, normalSum[1] / n, 0.02);
		assertEquals(2.0, normalProductSum[0] / n, 0.06);
		assertEquals(1.0, normalProductSum[1] / n, 0.04);
		assertEquals(0.6, normalProductSum[2] / n, 0.04);
		assertEquals(1.5, poissonSum[0] / n, 0.025);
		assertEquals(2.5, poissonSum[1] / n, 0.03);
		double poissonCovariance = poissonSum[2] / n -
				(poissonSum[0] / n) * (poissonSum[1] / n);
		assertEquals(0.5, poissonCovariance, 0.04);
	}

	@Test
	public void vectorRandomApisHaveExpectedShapes() {
		MersenneTwister random = new MersenneTwister(9L);
		int[] hyper = MultivariateHypergeometric.random(new int[] {5, 3, 2}, 4,
				random);
		assertEquals(4, hyper[0] + hyper[1] + hyper[2]);
		double[][] logistic = BivariateLogistic.random(2, 0, 1, 0, 1, random);
		int[][] counts = DirichletMultinomial.random(2, 4,
				new double[] {1, 2}, random);
		assertArrayEquals(new int[] {2, 2, 2, 2}, new int[] {logistic.length,
				logistic[0].length, counts.length, counts[0].length});
	}
}
