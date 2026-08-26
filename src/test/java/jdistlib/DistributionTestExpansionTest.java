/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.disttest.DistributionTest;
import jdistlib.rng.MersenneTwister;

public class DistributionTestExpansionTest {
	@Test
	public void oneSampleCramerVonMisesMatchesDefinition() {
		double[] sample = {0.1, 0.2, 0.3, 0.4, 0.5};
		double statistic = DistributionTest.cramer_von_mises_statistic(
				sample, new Uniform(0.0, 1.0));
		assertEquals(19.0 / 60.0, statistic, 1e-15);
	}

	@Test
	public void bootstrapGoodnessOfFitIsCallerReproducible() {
		double[] sample = {0.05, 0.12, 0.24, 0.48, 0.68, 0.81, 0.94};
		double[] first = DistributionTest.anderson_darling_test(sample,
				new Uniform(0.0, 1.0), 49, new MersenneTwister(42L));
		double[] second = DistributionTest.anderson_darling_test(sample,
				new Uniform(0.0, 1.0), 49, new MersenneTwister(42L));
		assertArrayEquals(first, second, 0.0);
		assertTrue(first[0] >= 0.0);
		assertTrue(first[1] > 0.0 && first[1] <= 1.0);
	}

	@Test
	public void twoSampleCramerVonMisesPermutationIsReproducible() {
		double[] firstSample = {1.0, 2.0, 3.0, 4.0};
		double[] secondSample = {5.0, 6.0, 7.0, 8.0};
		double[] first = DistributionTest.cramer_von_mises_test(firstSample,
				secondSample, 99, new MersenneTwister(17L));
		double[] second = DistributionTest.cramer_von_mises_test(firstSample,
				secondSample, 99, new MersenneTwister(17L));
		assertArrayEquals(first, second, 0.0);
		assertTrue(first[0] > 0.0);
		assertTrue(first[1] > 0.0 && first[1] <= 1.0);
	}

	@Test
	public void categoricalGoodnessOfFitReturnsStatisticPValueAndDf() {
		double[] result = DistributionTest.chi_square_goodness_of_fit_test(
				new long[] {20, 30, 50}, new double[] {0.25, 0.25, 0.5}, 0);
		assertEquals(2.0, result[0], 1e-15);
		assertEquals(Math.exp(-1.0), result[1], 1e-14);
		assertEquals(2.0, result[2], 0.0);
	}

	@Test
	public void contingencyIndependenceReturnsStatisticPValueAndDf() {
		double[] result = DistributionTest.chi_square_independence_test(
				new long[][] {{10, 20}, {20, 10}});
		assertEquals(20.0 / 3.0, result[0], 1e-14);
		assertEquals(ChiSquare.cumulative(20.0 / 3.0, 1.0, false, false),
				result[1], 1e-15);
		assertEquals(1.0, result[2], 0.0);
	}

	@Test
	public void newTestsRejectMalformedInputs() {
		assertThrows(IllegalArgumentException.class, () ->
				DistributionTest.cramer_von_mises_statistic(
						new double[] {Double.NaN, 1.0}, new Uniform(0.0, 1.0)));
		assertThrows(IllegalArgumentException.class, () ->
				DistributionTest.chi_square_goodness_of_fit_test(
						new long[] {2, 3}, new double[] {1.0, 0.0}, 0));
		assertThrows(IllegalArgumentException.class, () ->
				DistributionTest.chi_square_independence_test(
						new long[][] {{1, 2}, {0, 0}}));
	}
}
