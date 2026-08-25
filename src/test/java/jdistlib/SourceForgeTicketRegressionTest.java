/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.math.Bessel;

/** Focused regressions for the remaining historical SourceForge tickets. */
public class SourceForgeTicketRegressionTest {
	@Test
	public void ticketThirtyTracksTheCurrentNoncentralTUpperLogTail() {
		assertEquals(-3.7979583123021783,
				NonCentralT.cumulative(9.0, 7.5, 4.0, false, true), 2e-10);
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
	public void ticketThirteenRetainsNoncentralBetaDensityPrecision() {
		double expected = 3.001852308908624616864e-35;
		double actual = NonCentralBeta.density(0.8, 0.5, 5.0, 1000.0, false);
		assertEquals(expected, actual, expected * 2e-14);
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
	}

	@Test
	public void ticketNineKeepsTinyNoncentralChiSquareLogTailsFinite() {
		double logProbability = NonCentralChiSquare.cumulative(
				1e-5, 100.0, 1.0, true, true);
		assertTrue(Double.isFinite(logProbability));
		assertTrue(logProbability < -100.0);
	}

	@Test
	public void ticketFiveEnforcesHypergeometricIntegerSupport() {
		assertEquals(0.0, HyperGeometric.density(1.5, 10.0, 12.0, 5.0, false), 0.0);
		assertEquals(Double.NEGATIVE_INFINITY,
				HyperGeometric.density(1.5, 10.0, 12.0, 5.0, true), 0.0);
		assertFalse(Double.isNaN(HyperGeometric.density(
				2.0, 10.0 + 1e-12, 12.0, 5.0, false)));
	}

	@Test
	public void ticketOneKeepsHugeGammaLogTailsRepresentable() {
		double logProbability = Gamma.cumulative(
				0.9e100, 1e100, 1.0, true, true);
		assertTrue(Double.isFinite(logProbability));
		assertTrue(logProbability < -1e90);
	}
}
