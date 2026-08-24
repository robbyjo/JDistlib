/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class TweedieTest {
	private static final double MU = 1.4;
	private static final double PHI = 0.7;

	@Test
	public void specialCasesDelegateToTheirExactDistributions() {
		double y = 1.2;
		assertEquals(Normal.density(y, MU, Math.sqrt(PHI), false),
				Tweedie.density(y, MU, PHI, 0, false), 0);
		assertEquals(Poisson.density(2, 2, false) / PHI,
				Tweedie.density(1.4, MU, PHI, 1, false), 0);
		assertEquals(Gamma.density(y, 1 / PHI, PHI * MU, false),
				Tweedie.density(y, MU, PHI, 2, false), 0);
		assertEquals(InvNormal.density(y, MU, Math.sqrt(PHI), false),
				Tweedie.density(y, MU, PHI, 3, false), 0);
	}

	@Test
	public void compoundPoissonAndBigPowerMatchCranTweedie310() {
		assertEquals(0.034026453290862575, Tweedie.density(0, MU, PHI, 1.5, false), 2e-15);
		assertEquals(0.034026453290862575, Tweedie.cumulative(0, MU, PHI, 1.5, true, false), 2e-15);
		assertEquals(0.38231461607411732, Tweedie.density(1.2, MU, PHI, 1.5, false), 5e-12);
		assertEquals(0.50507596575023017, Tweedie.cumulative(1.2, MU, PHI, 1.5, true, false), 2e-12);
		assertEquals(0.36248200209437059, Tweedie.density(1.2, MU, PHI, 2.5, false), 7e-11);
		assertEquals(0.57432718937967775, Tweedie.cumulative(1.2, MU, PHI, 2.5, true, false), 2e-8);
	}

	@Test
	public void quantilesInvertTheCompletedCdf() {
		assertEquals(2.2100158586707468, Tweedie.quantile(0.8, MU, PHI, 1.5, true, false), 2e-10);
		assertEquals(2.1117177153302444, Tweedie.quantile(0.8, MU, PHI, 2.5, true, false), 2e-7);
		assertEquals(0.8, Tweedie.cumulative(
				Tweedie.quantile(Math.log(0.2), MU, PHI, 1.5, false, true),
				MU, PHI, 1.5, true, false), 2e-10);
	}

	@Test
	public void logLikelihoodAndUnitDevianceAreNoLongerPlaceholders() {
		assertEquals(Math.log(Tweedie.density(1.2, MU, PHI, 1.5, false)),
				Tweedie.loglik(1.2, MU, PHI, 1.5), 2e-15);
		assertEquals(0.030038368414580185, Tweedie.deviance(1.2, MU, 1), 2e-15);
		assertEquals(2 * MU, Tweedie.deviance(0, MU, 1), 0);
		assertEquals(-0.5 / PHI, Tweedie.dlogfdphi(MU, MU, PHI, 0), 0);
		assertEquals(0.65610615582434906, Tweedie.dlogfdphi(MU, MU, PHI, 1), 2e-12);
		assertEquals(-0.80844933107573702, Tweedie.dlogfdphi(MU, MU, PHI, 1.5), 2e-10);
		assertEquals(-0.79420483244975792, Tweedie.dlogfdphi(MU, MU, PHI, 2), 2e-12);
		assertEquals(-0.75487963814804471, Tweedie.dlogfdphi(MU, MU, PHI, 2.5), 2e-6);
		assertEquals(-0.71428571428571408, Tweedie.dlogfdphi(MU, MU, PHI, 3), 2e-7);
	}

	@Test
	public void randomCompoundPoissonHasTheRequestedMean() {
		Tweedie distribution = new Tweedie(MU, PHI, 1.5);
		distribution.setRandomEngine(new MersenneTwister(8675309));
		double sum = 0;
		int count = 20_000;
		for (int i = 0; i < count; i++) sum += distribution.random();
		assertEquals(MU, sum / count, 0.035);
		assertTrue(distribution.random() >= 0);
	}
}
