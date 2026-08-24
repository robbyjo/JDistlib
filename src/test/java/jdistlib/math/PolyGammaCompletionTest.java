/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.math;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;

public class PolyGammaCompletionTest {
	@Test
	public void negativeArgumentsMatchR461ThroughPentagamma() {
		double[] expected = {
			1.2668898312210739,
			10.661941561270993,
			30.814473885503073,
			359.26885865461713,
			3216.8447837603308,
			48696.490491429569
		};
		for (int order = 0; order < expected.length; order++)
			assertEquals(expected[order], PolyGamma.psigamma(-0.37, order),
					2e-12 * Math.max(1, Math.abs(expected[order])));
	}

	@Test
	public void negativeArgumentsWorkBeyondTheOldOrderThreeLimit() {
		double x = -0.37;
		double factorial = 1;
		for (int order = 0; order <= 12; order++) {
			if (order > 0) factorial *= order;
			double recurrence = ((order & 1) == 0 ? 1 : -1) * factorial
					/ Math.pow(x, order + 1);
			double expected = PolyGamma.psigamma(x + 1, order) - recurrence;
			double actual = PolyGamma.psigamma(x, order);
			assertEquals(expected, actual, 2e-11 * Math.max(1, Math.abs(expected)));
		}
	}

	@Test
	public void dpsifnReturnsTheRequestedSequenceLength() {
		double[] sequence = PolyGamma.dpsifn(-0.37, 4, 1, 5);
		assertNotNull(sequence);
		assertEquals(5, sequence.length);
		double factorial = 1;
		for (int k = 1; k <= 4; k++) factorial *= k;
		for (int j = 0; j < sequence.length; j++) {
			int order = 4 + j;
			if (j > 0) factorial *= order;
			double expected = ((order & 1) == 0 ? -1 : 1)
					* PolyGamma.psigamma(-0.37, order) / factorial;
			assertEquals(expected, sequence[j], 2e-11 * Math.max(1, Math.abs(expected)));
		}
	}
}
