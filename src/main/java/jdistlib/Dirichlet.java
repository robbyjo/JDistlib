/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Dirichlet distribution on a probability simplex. */
public final class Dirichlet {
	private Dirichlet() {}

	private static boolean validAlpha(double[] alpha) {
		if (alpha == null || alpha.length < 2) return false;
		for (double value : alpha)
			if (!(value > 0.0) || !Double.isFinite(value)) return false;
		return true;
	}

	public static double density(double[] x, double[] alpha, boolean giveLog) {
		if (!validAlpha(alpha) || x == null || x.length != alpha.length)
			return Double.NaN;
		double alphaSum = 0.0;
		double xSum = 0.0;
		double logDensity = 0.0;
		boolean zeroWithSmallShape = false;
		boolean zeroWithLargeShape = false;
		for (int i = 0; i < x.length; i++) {
			if (!(x[i] >= 0.0) || !Double.isFinite(x[i]))
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			xSum += x[i];
			alphaSum += alpha[i];
			logDensity -= lgammafn(alpha[i]);
			if (x[i] == 0.0) {
				if (alpha[i] < 1.0) zeroWithSmallShape = true;
				else if (alpha[i] > 1.0) zeroWithLargeShape = true;
			} else {
				logDensity += (alpha[i] - 1.0) * Math.log(x[i]);
			}
		}
		double tolerance = 16.0 * Math.ulp(1.0) * x.length;
		if (Math.abs(xSum - 1.0) > tolerance)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		if (zeroWithSmallShape && zeroWithLargeShape) return Double.NaN;
		if (zeroWithSmallShape)
			return Double.POSITIVE_INFINITY;
		if (zeroWithLargeShape)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		logDensity += lgammafn(alphaSum);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] alpha, RandomEngine random) {
		if (!validAlpha(alpha) || random == null) return null;
		double[] result = new double[alpha.length];
		for (int attempt = 0; attempt < 100; attempt++) {
			double sum = 0.0;
			for (int i = 0; i < alpha.length; i++) {
				result[i] = Gamma.random(alpha[i], 1.0, random);
				sum += result[i];
			}
			if (sum > 0.0 && Double.isFinite(sum)) {
				for (int i = 0; i < result.length; i++) result[i] /= sum;
				return result;
			}
		}
		return null;
	}

	public static double[][] random(int n, double[] alpha, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(alpha, random);
		return result;
	}
}
