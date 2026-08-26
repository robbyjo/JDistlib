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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.disttest.online.LordPlusPlus;
import jdistlib.disttest.online.OnlineFdr;
import jdistlib.disttest.online.OnlineFdrDecision;
import jdistlib.disttest.online.Saffron;

public class OnlineFdrTest {
	private static final double[] GAMMA = {0.5, 0.3, 0.2};

	@Test
	public void lordPlusPlusRewardsPastRejections() {
		LordPlusPlus lord = new LordPlusPlus(0.05, 0.025, GAMMA);
		OnlineFdrDecision first = lord.test(0.01);
		assertEquals(1, first.getIndex());
		assertEquals(0.0125, first.getTestLevel(), 1e-15);
		assertTrue(first.isRejected());
		OnlineFdrDecision second = lord.test(0.03);
		assertEquals(0.02, second.getTestLevel(), 1e-15);
		assertFalse(second.isRejected());
		assertEquals(0.0125, lord.test(0.5).getTestLevel(), 1e-15);
		assertEquals(3, lord.getTestCount());
		assertEquals(1, lord.getRejectionCount());
		lord.reset();
		assertEquals(0, lord.getTestCount());
	}

	@Test
	public void saffronTracksCandidatesAndRejectionRewards() {
		Saffron saffron = new Saffron(0.05, 0.01, 0.5, GAMMA);
		assertEquals(0.005, saffron.test(0.004).getTestLevel(), 1e-15);
		assertEquals(0.0125, saffron.test(0.8).getTestLevel(), 1e-15);
		OnlineFdrDecision third = saffron.test(0.006);
		assertEquals(0.0075, third.getTestLevel(), 1e-15);
		assertTrue(third.isRejected());
		assertEquals(2, saffron.getCandidateCount());
		assertEquals(0.02, saffron.test(0.9).getTestLevel(), 1e-15);
		assertEquals(2, saffron.getRejectionCount());
	}

	@Test
	public void onlineInputsAndGammaContractsAreValidated() {
		assertEquals(1.0, sum(OnlineFdr.polynomialGamma(100, 1.6)), 1e-15);
		assertThrows(IllegalArgumentException.class, () ->
				new LordPlusPlus(0.05, 0.06, GAMMA));
		assertThrows(IllegalArgumentException.class, () ->
				new LordPlusPlus(0.05, 0.025,
						new double[] {0.4, 0.5}));
		assertThrows(IllegalArgumentException.class, () ->
				new Saffron(0.05, 0.025, 0.5, GAMMA));
		LordPlusPlus lord = new LordPlusPlus(0.05, 0.025, GAMMA);
		assertThrows(IllegalArgumentException.class, () -> lord.test(Double.NaN));
	}

	private static double sum(double[] values) {
		double sum = 0.0;
		for (double value : values) sum += value;
		return sum;
	}
}
