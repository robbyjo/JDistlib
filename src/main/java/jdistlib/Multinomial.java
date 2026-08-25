/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Multinomial mass and random generation for a vector of category counts. */
public final class Multinomial {
	private Multinomial() {}

	private static double[] probabilities(double[] weights) {
		if (weights == null || weights.length == 0) return null;
		double sum = 0.0;
		for (double weight : weights) {
			if (!(weight >= 0.0) || Double.isInfinite(weight)) return null;
			sum += weight;
		}
		if (!(sum > 0.0) || Double.isInfinite(sum)) return null;
		double[] result = weights.clone();
		for (int i = 0; i < result.length; i++) result[i] /= sum;
		return result;
	}

	public static double density(double[] x, int size, double[] weights,
			boolean giveLog) {
		double[] p = probabilities(weights);
		if (x == null || p == null || x.length != p.length || size < 0) {
			return Double.NaN;
		}
		double total = 0.0;
		double result = lgammafn(size + 1.0);
		for (int i = 0; i < x.length; i++) {
			if (x[i] < 0.0 || x[i] != Math.rint(x[i]) || Double.isInfinite(x[i])) {
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			}
			total += x[i];
			if (p[i] == 0.0 && x[i] > 0.0) {
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			}
			result -= lgammafn(x[i] + 1.0);
			if (x[i] > 0.0) result += x[i] * Math.log(p[i]);
		}
		if (total != size) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		return giveLog ? result : Math.exp(result);
	}

	public static int[] random(int size, double[] weights, RandomEngine random) {
		double[] p = probabilities(weights);
		if (p == null || size < 0) return null;
		double[] cumulative = new double[p.length];
		cumulative[0] = p[0];
		for (int i = 1; i < p.length; i++) cumulative[i] = cumulative[i - 1] + p[i];
		int[] result = new int[p.length];
		for (int draw = 0; draw < size; draw++) {
			double u = random.nextDouble();
			int category = 0;
			while (category + 1 < cumulative.length && u > cumulative[category]) {
				category++;
			}
			result[category]++;
		}
		return result;
	}

	public static int[][] random(int n, int size, double[] weights,
			RandomEngine random) {
		int[][] result = new int[n][];
		for (int i = 0; i < n; i++) result[i] = random(size, weights, random);
		return result;
	}
}
