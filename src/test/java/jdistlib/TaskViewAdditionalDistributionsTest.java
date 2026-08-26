/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class TaskViewAdditionalDistributionsTest {
	@Test
	public void asymmetricLaplaceMatchesItsSymmetricParentAndQuantileDefinition() {
		for (double x : new double[] {-10.0, -1.0, 0.0, 0.7, 8.0}) {
			assertEquals(Laplace.density(x, 0.0, 2.6, false),
					AsymmetricLaplace.density(x, 0.0, 1.3, 0.5, false), 3e-16);
			assertEquals(Laplace.cumulative(x, 0.0, 2.6, true, false),
					AsymmetricLaplace.cumulative(x, 0.0, 1.3, 0.5,
							true, false), 3e-16);
		}
		assertEquals(4.2, AsymmetricLaplace.quantile(0.73, 4.2, 1.7,
				0.73, true, false), 0.0);
	}

	@Test
	public void huberMatchesItsPiecewiseDefinitionAndInvertsTails() {
		double c = 1.0;
		double phiC = Normal.density(c, 0.0, 1.0, false);
		double area = 2.0 * (Normal.cumulative(c, 0.0, 1.0, true, false)
				+ phiC / c - 0.5);
		assertEquals(Normal.density(0.0, 0.0, 1.0, false) / area,
				Huber.density(0.0, 0.0, 1.0, c, false), 2e-16);
		assertEquals(0.5, Huber.cumulative(0.0, 0.0, 1.0, c,
				true, false), 0.0);
		assertEquals(Huber.cumulative(-2.3, 0.0, 1.0, c, true, false),
				Huber.cumulative(2.3, 0.0, 1.0, c, false, false), 0.0);
		for (double p : new double[] {1e-12, 0.01, 0.2, 0.5, 0.8, 0.99,
				1.0 - 1e-12}) {
			double value = Huber.quantile(p, -0.4, 1.7, 1.2, true, false);
			assertEquals(p, Huber.cumulative(value, -0.4, 1.7, 1.2,
					true, false), 2e-13);
		}
	}

	@Test
	public void exponentiallyModifiedGaussianMatchesCranFormulaAndInverts() {
		double correction = Math.exp(0.5)
				* Normal.cumulative(-1.0, 0.0, 1.0, true, false);
		assertEquals(correction,
				ExponentiallyModifiedGaussian.density(0.0, 0.0, 1.0, 1.0,
						false), 3e-16);
		assertEquals(0.5 - correction,
				ExponentiallyModifiedGaussian.cumulative(0.0, 0.0, 1.0, 1.0,
						true, false), 3e-16);
		for (double p : new double[] {1e-10, 0.001, 0.1, 0.5, 0.9, 0.999,
				1.0 - 1e-10}) {
			double value = ExponentiallyModifiedGaussian.quantile(p, -0.7, 1.2,
					0.8, true, false);
			assertEquals(p, ExponentiallyModifiedGaussian.cumulative(value, -0.7,
					1.2, 0.8, true, false), 2e-12);
		}
	}

	@Test
	public void loggedUpperTailsInvertForAllThreeLaws() {
		GenericDistribution[] distributions = {
			new AsymmetricLaplace(-0.2, 1.3, 0.7),
			new Huber(0.4, 1.5, 1.345),
			new ExponentiallyModifiedGaussian(-0.7, 1.2, 0.8)
		};
		for (GenericDistribution distribution : distributions) {
			double logP = Math.log(1e-8);
			double value = distribution.quantile(logP, false, true);
			assertEquals(logP, distribution.cumulative(value, false, true), 2e-9);
		}
	}

	@Test
	public void randomGeneratorsRecoverTheirMeans() {
		MersenneTwister random = new MersenneTwister(20260826L);
		int n = 30000;
		double asymmetric = 0.0;
		double huber = 0.0;
		double emg = 0.0;
		for (int i = 0; i < n; i++) {
			asymmetric += AsymmetricLaplace.random(2.0, 1.0, 0.5, random);
			huber += Huber.random(-1.0, 1.3, 1.345, random);
			emg += ExponentiallyModifiedGaussian.random(-0.5, 1.2, 0.8, random);
		}
		assertEquals(2.0, asymmetric / n, 0.035);
		assertEquals(-1.0, huber / n, 0.025);
		assertEquals(-0.5 + 1.0 / 0.8, emg / n, 0.03);
	}
}
