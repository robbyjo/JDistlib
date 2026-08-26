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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import jdistlib.disttest.MultipleTesting;
import jdistlib.disttest.MultipleTesting.AdaptiveFdrResult;
import jdistlib.disttest.MultipleTesting.CensoredTestResult;
import jdistlib.disttest.MultipleTesting.Method;

public class MultipleTestingTest {
	private static final double[] P = {0.01, 0.04, 0.03, 0.002, 0.5,
			Double.NaN};

	@Test
	public void standardAdjustmentsMatchReferenceValues() {
		assertArrayEquals(new double[] {0.05, 0.20, 0.15, 0.01, 1.0,
				Double.NaN}, MultipleTesting.adjust(P, Method.BONFERRONI), 1e-15);
		assertArrayEquals(new double[] {0.04, 0.09, 0.09, 0.01, 0.5,
				Double.NaN}, MultipleTesting.adjust(P, Method.HOLM), 1e-15);
		assertArrayEquals(new double[] {0.04, 0.08, 0.08, 0.01, 0.5,
				Double.NaN}, MultipleTesting.adjust(P, Method.HOCHBERG), 1e-15);
		assertArrayEquals(new double[] {0.025, 0.05, 0.05, 0.01, 0.5,
				Double.NaN}, MultipleTesting.adjust(P,
				Method.BENJAMINI_HOCHBERG), 1e-15);

		double harmonic = 1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0
				+ 1.0 / 5.0;
		assertArrayEquals(new double[] {0.025 * harmonic, 0.05 * harmonic,
				0.05 * harmonic, 0.01 * harmonic, 1.0,
				Double.NaN}, MultipleTesting.adjust(P,
				Method.BENJAMINI_YEKUTIELI), 1e-15);
	}

	@Test
	public void sidakMethodsUseStableStandardDirection() {
		double[] sidak = MultipleTesting.adjust(P, Method.SIDAK);
		assertEquals(-Math.expm1(5.0 * Math.log1p(-0.01)), sidak[0], 1e-15);
		assertEquals(-Math.expm1(5.0 * Math.log1p(-0.5)), sidak[4], 1e-15);

		double[] holmSidak = MultipleTesting.adjust(P, Method.HOLM_SIDAK);
		assertEquals(-Math.expm1(4.0 * Math.log1p(-0.01)),
				holmSidak[0], 1e-15);
		assertEquals(-Math.expm1(3.0 * Math.log1p(-0.03)),
				holmSidak[1], 1e-15);
		assertEquals(holmSidak[1], holmSidak[2], 1e-15);
	}

	@Test
	public void hommelMatchesIndependentClosedSimesEnumeration() {
		double[] input = {0.011, 0.042, 0.018, 0.23, 0.61, 0.003};
		double[] expected = bruteForceHommel(input);
		assertArrayEquals(expected,
				MultipleTesting.adjust(input, Method.HOMMEL), 1e-15);
	}

	@Test
	public void thresholdAndRejectionHelpersUseAdjustedValues() {
		assertEquals(0.04, MultipleTesting.threshold(P, 0.05,
				Method.BENJAMINI_HOCHBERG), 0.0);
		assertEquals(4, MultipleTesting.countRejected(P, 0.05,
				Method.BENJAMINI_HOCHBERG));
		boolean[] rejected = MultipleTesting.reject(P, 0.05,
				Method.BENJAMINI_HOCHBERG);
		assertTrue(rejected[0]);
		assertTrue(rejected[1]);
		assertFalse(rejected[4]);
		assertFalse(rejected[5]);
		assertTrue(Double.isNaN(MultipleTesting.threshold(
				new double[] {0.4, 0.8}, 0.05, Method.HOLM)));
	}

	@Test
	public void explicitFamilySizeIsSupported() {
		double[] pValues = {0.01, 0.04};
		assertArrayEquals(new double[] {0.10, 0.36},
				MultipleTesting.adjust(pValues, Method.HOLM, 10), 1e-15);
		assertArrayEquals(new double[] {0.10, 0.20},
				MultipleTesting.adjust(pValues,
						Method.BENJAMINI_HOCHBERG, 10), 1e-15);
	}

	@Test
	public void logAdjustmentsMatchOrdinaryAdjustments() {
		double[] pValues = {1e-200, 0.002, 0.01, 0.12, 0.8, Double.NaN};
		double[] logPValues = new double[pValues.length];
		for (int i = 0; i < pValues.length; i++)
			logPValues[i] = Double.isNaN(pValues[i]) ? Double.NaN
					: Math.log(pValues[i]);
		for (Method method : Method.values()) {
			double[] expected = MultipleTesting.adjust(pValues, method);
			double[] actualLog = MultipleTesting.adjustLog(logPValues, method);
			for (int i = 0; i < expected.length; i++) {
				if (Double.isNaN(expected[i])) assertTrue(Double.isNaN(actualLog[i]));
				else assertEquals(expected[i], Math.exp(actualLog[i]), 2e-15);
			}
		}
	}

	@Test
	public void logAdjustmentsPreserveValuesBelowProbabilityUnderflow() {
		double[] adjusted = MultipleTesting.adjustLog(
				new double[] {-1000.0, -900.0}, Method.BONFERRONI);
		assertEquals(-1000.0 + Math.log(2.0), adjusted[0], 0.0);
		assertEquals(-900.0 + Math.log(2.0), adjusted[1], 0.0);
		assertTrue(MultipleTesting.rejectLog(new double[] {-1000.0},
				1e-300, Method.BONFERRONI)[0]);
	}

	@Test
	public void bkyReportsBothLevelDependentStages() {
		double[] pValues = {0.001, 0.004, 0.01, 0.03, 0.10, 0.80};
		AdaptiveFdrResult result = MultipleTesting
				.benjaminiKriegerYekutieli(pValues, 0.05);
		assertEquals(4, result.getStageOneRejections());
		assertEquals(2, result.getEstimatedTrueNulls());
		assertEquals(0.05 / 1.05, result.getStageOneLevel(), 1e-15);
		assertEquals((0.05 / 1.05) * 6.0 / 2.0,
				result.getFinalLevel(), 1e-15);
		assertEquals(5, result.getRejectedCount());
		assertEquals(0.10, result.getThreshold(), 0.0);
		assertFalse(result.getRejected()[5]);
	}

	@Test
	public void rightCensoredFamiliesUseKnownTotalConservatively() {
		double[] recorded = {0.001, 0.004, 0.01, 0.03};
		CensoredTestResult result = MultipleTesting.testRightCensored(
				recorded, 0.05, 6, 0.05, Method.BENJAMINI_HOCHBERG);
		assertArrayEquals(new double[] {0.006, 0.012, 0.02, 0.045},
				result.getAdjustedPValues(), 1e-15);
		assertEquals(2, result.getUnobservedCount());
		assertEquals(4, result.getRejectedCount());
		assertTrue(result.areDecisionsExact());

		CensoredTestResult uncertain = MultipleTesting.testRightCensored(
				new double[] {0.001, 0.009}, 0.01, 10, 0.05,
				Method.BENJAMINI_HOCHBERG);
		assertFalse(uncertain.areDecisionsExact());
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.testRightCensored(new double[] {0.02}, 0.01,
						10, 0.05, Method.BENJAMINI_HOCHBERG));
	}

	@Test
	public void qValuesWithPiZeroOneEqualBenjaminiHochberg() {
		assertArrayEquals(MultipleTesting.adjust(P,
				Method.BENJAMINI_HOCHBERG), MultipleTesting.qValues(P, 1.0),
				1e-15);
	}

	@Test
	public void quantileNullEstimateMatchesHandCalculation() {
		double[] pValues = {0.01, 0.20, 0.40, 0.60, 0.80};
		double pi0 = MultipleTesting.estimateNullProportionQuantile(pValues,
				new double[] {0.0, 0.5}, 0.5);
		assertEquals(0.9, pi0, 1e-15);
		double[] qValues = MultipleTesting.qValues(pValues, pi0);
		for (int i = 1; i < qValues.length; i++)
			assertTrue(qValues[i] >= qValues[i - 1]);
	}

	@Test
	public void splineEstimateAndFdrAreFiniteProbabilities() {
		double[] pValues = new double[200];
		for (int i = 0; i < pValues.length; i++)
			pValues[i] = (i + 0.5) / pValues.length;
		double pi0 = MultipleTesting.estimateNullProportion(pValues);
		assertTrue(pi0 >= 0.0 && pi0 <= 1.0);
		double fdr = MultipleTesting.estimatedFalseDiscoveryRate(
				pValues, 0.05, pi0);
		assertTrue(fdr >= 0.0 && fdr <= 1.0);
	}

	@Test
	public void inputsAreNotMutatedAndInvalidValuesAreRejected() {
		double[] copy = P.clone();
		MultipleTesting.adjust(P, Method.HOLM);
		assertArrayEquals(copy, P, 0.0);
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.adjust(new double[] {-0.1}, Method.HOLM));
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.adjust(new double[] {1.1}, Method.HOLM));
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.adjust(new double[] {0.1, 0.2}, Method.HOLM, 1));
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.qValues(new double[] {0.1}, 1.1));
		assertThrows(IllegalArgumentException.class, () ->
				MultipleTesting.adjustLog(new double[] {0.01}, Method.HOLM));
		assertThrows(ArithmeticException.class, () ->
				MultipleTesting.estimateNullProportionQuantile(
						new double[] {0.001, 0.002, 0.003},
						new double[] {0.5}, 0.5));
	}

	private static double[] bruteForceHommel(double[] pValues) {
		int count = pValues.length;
		double[] result = new double[count];
		for (int hypothesis = 0; hypothesis < count; hypothesis++) {
			double maximum = 0.0;
			for (int subset = 1; subset < (1 << count); subset++) {
				if ((subset & (1 << hypothesis)) == 0) continue;
				double[] values = new double[Integer.bitCount(subset)];
				int next = 0;
				for (int i = 0; i < count; i++) {
					if ((subset & (1 << i)) != 0) values[next++] = pValues[i];
				}
				Arrays.sort(values);
				double simes = 1.0;
				for (int rank = 0; rank < values.length; rank++) {
					simes = Math.min(simes,
							values.length * values[rank] / (rank + 1.0));
				}
				maximum = Math.max(maximum, simes);
			}
			result[hypothesis] = maximum;
		}
		return result;
	}
}
