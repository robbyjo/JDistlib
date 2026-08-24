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

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class AdditionalDistributionsTest {
	private static final double[] X = {-1.0, 0.0, 1.25, 3.0,
			Double.POSITIVE_INFINITY};

	private static void assertVector(double[] expected, double[] actual,
			double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) {
			assertEquals("element " + i, expected[i], actual[i], tolerance);
		}
	}

	@Test
	public void triangularMatchesExtraDistr11005() {
		double[] density = new double[X.length];
		double[] cumulative = new double[X.length];
		double[] upperLog = new double[X.length];
		for (int i = 0; i < X.length; i++) {
			density[i] = Triangular.density(X[i], -2.0, 5.0, 1.5, false);
			cumulative[i] = Triangular.cumulative(X[i], -2.0, 5.0, 1.5,
					true, false);
			upperLog[i] = Triangular.cumulative(X[i], -2.0, 5.0, 1.5,
					false, true);
		}
		assertVector(new double[] {0.0816326530612245, 0.163265306122449,
				0.265306122448980, 0.163265306122449, 0.0}, density, 5e-15);
		assertVector(new double[] {0.0408163265306122, 0.163265306122449,
				0.431122448979592, 0.836734693877551, 1.0}, cumulative, 5e-15);
		assertVector(new double[] {-0.0416726964005681, -0.178248231406319,
				-0.564090068330344, -1.81237875643079,
				Double.NEGATIVE_INFINITY}, upperLog, 5e-14);
		assertVector(new double[] {-2.0, -1.50502525316942, 0.213594362117866,
				1.5, 3.43475241575015, 5.0}, triangularQuantiles(), 5e-14);

		// Endpoint modes are valid triangular distributions.
		assertEquals(2.0 / 7.0, Triangular.density(-2.0, -2.0, 5.0, -2.0,
				false), 0.0);
		assertEquals(5.0, Triangular.quantile(1.0, -2.0, 5.0, 5.0, true,
				false), 0.0);
	}

	private static double[] triangularQuantiles() {
		double[] probabilities = {0.0, 0.01, 0.2, 0.5, 0.9, 1.0};
		double[] result = new double[probabilities.length];
		for (int i = 0; i < probabilities.length; i++) {
			result[i] = Triangular.quantile(probabilities[i], -2.0, 5.0, 1.5,
					true, false);
		}
		return result;
	}

	@Test
	public void halfNormalMatchesExtraDistr11005() {
		double[] density = new double[X.length];
		double[] cumulative = new double[X.length];
		double[] upperLog = new double[X.length];
		for (int i = 0; i < X.length; i++) {
			density[i] = HalfNormal.density(X[i], 2.3, false);
			cumulative[i] = HalfNormal.cumulative(X[i], 2.3, true, false);
			upperLog[i] = HalfNormal.cumulative(X[i], 2.3, false, true);
		}
		assertVector(new double[] {0.0, 0.346906330783854, 0.299277277565029,
				0.148174877017693, 0.0}, density, 5e-15);
		assertVector(new double[] {0.0, 0.0, 0.413199445347418,
				0.807884984420455, 1.0}, cumulative, 5e-15);
		assertVector(new double[] {0.0, 0.0, -0.533070287498041,
				-1.64966104683008, Double.NEGATIVE_INFINITY}, upperLog, 5e-14);
		assertVector(new double[] {0.0, 0.0288269798685593, 0.582698337212339,
				1.55132642545099, 3.78316334198838,
				Double.POSITIVE_INFINITY}, halfNormalQuantiles(), 5e-14);
	}

	private static double[] halfNormalQuantiles() {
		double[] probabilities = {0.0, 0.01, 0.2, 0.5, 0.9, 1.0};
		double[] result = new double[probabilities.length];
		for (int i = 0; i < probabilities.length; i++) {
			result[i] = HalfNormal.quantile(probabilities[i], 2.3, true, false);
		}
		return result;
	}

	@Test
	public void birnbaumSaundersMatchesExtraDistr11005() {
		double[] density = new double[X.length];
		double[] cumulative = new double[X.length];
		double[] upperLog = new double[X.length];
		for (int i = 0; i < X.length; i++) {
			density[i] = BirnbaumSaunders.density(X[i], 0.7, 2.1, -0.5, false);
			cumulative[i] = BirnbaumSaunders.cumulative(X[i], 0.7, 2.1, -0.5,
					true, false);
			upperLog[i] = BirnbaumSaunders.cumulative(X[i], 0.7, 2.1, -0.5,
					false, true);
		}
		assertVector(new double[] {0.0, 0.120151779038871, 0.316085168469516,
				0.128109983138657, 0.0}, density, 5e-15);
		assertVector(new double[] {0.0, 0.0128528657792565, 0.397115557813063,
				0.769654991394511, 1.0}, cumulative, 5e-15);
		assertVector(new double[] {0.0, -0.0129361784999976, -0.506029738784174,
				-1.46817705657608, Double.NEGATIVE_INFINITY}, upperLog, 5e-14);
		assertVector(new double[] {-0.5, -0.0255185979019633,
				0.674691908506675, 1.6, 4.50971339360687,
				Double.POSITIVE_INFINITY}, birnbaumSaundersQuantiles(), 8e-14);
	}

	private static double[] birnbaumSaundersQuantiles() {
		double[] probabilities = {0.0, 0.01, 0.2, 0.5, 0.9, 1.0};
		double[] result = new double[probabilities.length];
		for (int i = 0; i < probabilities.length; i++) {
			result[i] = BirnbaumSaunders.quantile(probabilities[i], 0.7, 2.1,
					-0.5, true, false);
		}
		return result;
	}

	@Test
	public void gompertzMatchesFlexsurv232AndSupportsNegativeShape() {
		double[] x = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
		double[] expected = {0.0, 0.2, 0.179105591827508, 0.156884811322895,
				0.134101872197705, 0.111571759992743};
		for (int i = 0; i < x.length; i++) {
			assertEquals(expected[i], Gompertz.density(x[i], 0.1, 0.2, false),
					5e-15);
		}

		assertEquals(1.28150707286845,
				Gompertz.quantile(0.8, -0.6, 1.8, true, false), 5e-14);
		assertEquals(2.4316450975351,
				Gompertz.quantile(0.9, -0.6, 1.8, true, false), 5e-14);
		assertEquals(Double.POSITIVE_INFINITY,
				Gompertz.quantile(0.97, -0.6, 1.8, true, false), 0.0);
		assertEquals(Math.exp(-3.0),
				Gompertz.cumulative(Double.POSITIVE_INFINITY, -0.6, 1.8,
						false, false), 5e-16);
		assertEquals(Math.exp(-0.2),
				Gompertz.cumulative(Double.MAX_VALUE, -1.0, 0.2,
						false, false), 5e-16);
		assertEquals(Exponential.cumulative(2.0, 5.0, true, false),
				Gompertz.cumulative(2.0, 0.0, 0.2, true, false), 0.0);
		assertEquals(0.2 * Math.exp(-0.4 * 3.0),
				Gompertz.hazard(3.0, -0.4, 0.2, false), 2e-17);
	}

	@Test
	public void quantilesInvertBothLoggedTailsThroughInstanceApi() {
		GenericDistribution[] distributions = {
			new Triangular(-2.0, 5.0, 1.5),
			new HalfNormal(2.3),
			new BirnbaumSaunders(0.7, 2.1, -0.5),
			new Gompertz(0.13, 0.7)
		};
		for (GenericDistribution distribution : distributions) {
			double probability = 1e-10;
			double value = distribution.quantile(Math.log(probability), false, true);
			assertEquals(Math.log(probability),
					distribution.cumulative(value, false, true), 2e-6);
		}
	}

	@Test
	public void randomGeneratorsHaveExpectedMomentsAndDefectiveMass() {
		int count = 30_000;
		MersenneTwister random = new MersenneTwister(20260824L);
		double triangularSum = 0.0;
		double halfNormalSum = 0.0;
		double fatigueSum = 0.0;
		int infiniteGompertz = 0;
		for (int i = 0; i < count; i++) {
			triangularSum += Triangular.random(-2.0, 5.0, 1.5, random);
			halfNormalSum += HalfNormal.random(2.3, random);
			fatigueSum += BirnbaumSaunders.random(0.7, 2.1, -0.5, random);
			if (Double.isInfinite(Gompertz.random(-0.4, 0.2, random))) {
				infiniteGompertz++;
			}
		}
		assertEquals(1.5, triangularSum / count, 0.025);
		assertEquals(2.3 * Math.sqrt(2.0 / Math.PI), halfNormalSum / count, 0.03);
		assertEquals(-0.5 + 2.1 * (1.0 + 0.7 * 0.7 / 2.0),
				fatigueSum / count, 0.05);
		assertEquals(Math.exp(-0.5), infiniteGompertz / (double) count, 0.015);
		assertTrue(infiniteGompertz > 0);
	}
}
