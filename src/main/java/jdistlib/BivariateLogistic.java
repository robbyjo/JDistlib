/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Gumbel's type-I bivariate logistic distribution as used by VGAM. */
public final class BivariateLogistic {
	private BivariateLogistic() {}

	private static boolean valid(double location1, double scale1,
			double location2, double scale2) {
		return Double.isFinite(location1) && Double.isFinite(location2) &&
				scale1 > 0.0 && scale2 > 0.0 && Double.isFinite(scale1) &&
				Double.isFinite(scale2);
	}

	private static double logDenominator(double z1, double z2) {
		double a = -z1;
		double b = -z2;
		double maximum = Math.max(0.0, Math.max(a, b));
		return maximum + Math.log(Math.exp(-maximum) + Math.exp(a - maximum) +
				Math.exp(b - maximum));
	}

	public static double density(double x1, double x2, double location1,
			double scale1, double location2, double scale2, boolean giveLog) {
		if (!valid(location1, scale1, location2, scale2)) return Double.NaN;
		if (Double.isNaN(x1) || Double.isNaN(x2)) return x1 + x2;
		if (!Double.isFinite(x1) || !Double.isFinite(x2))
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		double z1 = (x1 - location1) / scale1;
		double z2 = (x2 - location2) / scale2;
		double logDensity = Math.log(2.0) - z1 - z2 - Math.log(scale1) -
				Math.log(scale2) - 3.0 * logDenominator(z1, z2);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	/** Returns {@code P[X1 <= q1, X2 <= q2]}. */
	public static double cumulative(double q1, double q2, double location1,
			double scale1, double location2, double scale2, boolean logP) {
		if (!valid(location1, scale1, location2, scale2)) return Double.NaN;
		if (Double.isNaN(q1) || Double.isNaN(q2)) return q1 + q2;
		if (q1 == Double.NEGATIVE_INFINITY || q2 == Double.NEGATIVE_INFINITY)
			return logP ? Double.NEGATIVE_INFINITY : 0.0;
		double z1 = (q1 - location1) / scale1;
		double z2 = (q2 - location2) / scale2;
		double logProbability = -logDenominator(z1, z2);
		return logP ? logProbability : Math.exp(logProbability);
	}

	public static double[] random(double location1, double scale1,
			double location2, double scale2, RandomEngine random) {
		if (!valid(location1, scale1, location2, scale2) || random == null) return null;
		double y1 = Logistic.random(location1, scale1, random);
		double expNegativeZ1 = Math.exp(-(y1 - location1) / scale1);
		double u = random.nextDouble();
		while (u == 0.0) u = random.nextDouble();
		double inside = 1.0 / Math.sqrt(u / ((1.0 + expNegativeZ1) *
				(1.0 + expNegativeZ1))) - 1.0 - expNegativeZ1;
		double y2 = location2 - scale2 * Math.log(inside);
		return new double[] {y1, y2};
	}

	public static double[][] random(int n, double location1, double scale1,
			double location2, double scale2, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++)
			result[i] = random(location1, scale1, location2, scale2, random);
		return result;
	}
}
