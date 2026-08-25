/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.abs;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.math.Bessel;
import jdistlib.rng.MersenneTwister;

/** Focused regressions for the remaining historical SourceForge tickets. */
public class SourceForgeTicketRegressionTest {
	@Test
	public void ticketThirtyRemovesThePr16845NoncentralTDiscontinuity() {
		assertEquals(-3.7979583123021783,
				NonCentralT.cumulative(9.0, 7.5, 4.0, false, true), 2e-10);

		double[][] cases = {
				{0.0378, 3.75, 0.15429891097300658974488321321376159},
				{0.0378, 3.76, 0.15539778452845913495491252761260947},
				{0.02, 3.759, 0.45187612058802224726627608785540290},
				{0.02, 3.7591, 0.45188815299772598689884861170274824}
		};
		double[] actual = new double[cases.length];
		for (int i = 0; i < cases.length; i++) {
			double[] current = cases[i];
			double t = 1.0 / current[0];
			double ncp = sqrt(2.0) * 100.0 / current[1];
			actual[i] = NonCentralT.cumulative(
					t, 1.0, ncp, true, false);
			assertEquals(current[2], actual[i], 2e-12);
			assertEquals(log(current[2]), NonCentralT.cumulative(
					t, 1.0, ncp, true, true), 1e-11);
		}
		assertTrue(abs(actual[1] - actual[0]) < 0.002);
		assertTrue(abs(actual[3] - actual[2]) < 0.00002);
	}

	@Test
	public void ticketTwentyThreeRetainsPbetaLogTailPrecision() {
		assertEquals(-994.769290541658,
				Beta.cumulative(0.555555, 1925.74, 33.7179, true, true), 2e-10);
		assertEquals(-994.767594138466967,
				Beta.cumulative(0.5555555, 1925.74, 33.7179, true, true), 2e-10);
		assertEquals(-994.769290541658,
				Beta.cumulative(0.444445, 33.7179, 1925.74, false, true), 2e-10);
		assertEquals(-993.424624967607243,
				Beta.cumulative(5.0 / 9.0, 1925.0, 34.0, true, true), 2e-10);
	}

	@Test
	public void ticketThirteenUsesCompensatedNoncentralBetaDensityArithmetic() {
		/* 100-decimal Poisson/beta mixture evaluated at the exact binary64 inputs. */
		double expected = 3.00185230890896372644330235739443699e-35;
		double actual = NonCentralBeta.density(0.8, 0.5, 5.0, 1000.0, false);
		assertEquals(expected, actual, expected * 3e-15);
		assertEquals(-79.4912487203558705288943536530957356,
				NonCentralBeta.density(0.8, 0.5, 5.0, 1000.0, true), 2e-14);
	}

	@Test
	public void ticketTwelveRejectsUnrepresentableBesselOrdersWithoutAllocation() {
		assertTrue(Double.isNaN(Bessel.j(1.0, Math.scalb(1.0, 64))));
		assertTrue(Double.isNaN(Bessel.y(1.0, Math.scalb(1.0, 64))));
	}

	@Test
	public void ticketElevenRetainsPoissonDensityPrecisionNearAHugeMean() {
		double expected = 5.520992859342143e-98;
		double actual = Poisson.density(1e20 - 2e11, 1e20, false);
		assertEquals(expected, actual, expected * 1e-13);
		assertEquals(-3.2745662778118176,
				Poisson.density(5.0, 10.0, true), 0.0);
	}

	@Test
	public void ticketNineUsesStableNoncentralChiSquareTailsForInversion() {
		double logProbability = NonCentralChiSquare.cumulative(
				1e-5, 100.0, 1.0, true, true);
		assertTrue(Double.isFinite(logProbability));
		assertTrue(logProbability < -100.0);

		double[][] cases = {
				{38.1, 0.1}, {38.5, 0.5}, {39.5, 1.5},
				{39.7, 4.7}, {42.7, 4.7}
		};
		for (double[] current : cases) {
			double x = current[0];
			double df = current[1];
			double lower = NonCentralChiSquare.cumulative(x, df, 1.0, true, false);
			double lowerLog = NonCentralChiSquare.cumulative(x, df, 1.0, true, true);
			double upper = NonCentralChiSquare.cumulative(x, df, 1.0, false, false);
			assertEquals(x,
					NonCentralChiSquare.quantile(lower, df, 1.0, true, false), 5e-9);
			assertEquals(x,
					NonCentralChiSquare.quantile(lowerLog, df, 1.0, true, true), 2e-12);
			assertEquals(x,
					NonCentralChiSquare.quantile(upper, df, 1.0, false, false), 2e-12);
		}
		assertTrue(Double.isFinite(NonCentralChiSquare.quantile(
				-1e-20, 4.7, 1.0, true, true)));
	}

	@Test
	public void ticketFiveEnforcesHypergeometricIntegerParametersAndSupport() {
		assertEquals(0.0, HyperGeometric.density(1.5, 10.0, 12.0, 5.0, false), 0.0);
		assertEquals(Double.NEGATIVE_INFINITY,
				HyperGeometric.density(1.5, 10.0, 12.0, 5.0, true), 0.0);
		assertFalse(Double.isNaN(HyperGeometric.density(
				2.0, 10.0 + 1e-12, 12.0, 5.0, false)));

		assertTrue(Double.isNaN(HyperGeometric.density(
				2.0, 10.4, 12.0, 5.0, false)));
		assertTrue(Double.isNaN(HyperGeometric.cumulative(
				2.0, 10.4, 12.0, 5.0, true, false)));
		assertTrue(Double.isNaN(HyperGeometric.quantile(
				0.5, 10.4, 12.0, 5.0, true, false)));
		assertTrue(Double.isNaN(HyperGeometric.random(
				10.4, 12.0, 5.0, new MersenneTwister(7393L))));
	}

	@Test
	public void ticketOneKeepsHugeGammaLogTailsRepresentable() {
		double logProbability = Gamma.cumulative(
				0.9e100, 1e100, 1.0, true, true);
		assertTrue(Double.isFinite(logProbability));
		assertTrue(logProbability < -1e90);
	}
}
