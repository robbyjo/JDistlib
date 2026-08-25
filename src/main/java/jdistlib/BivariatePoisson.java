/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Bivariate Poisson distribution formed from three independent Poisson counts. */
public final class BivariatePoisson {
	private BivariatePoisson() {}

	private static boolean valid(double lambda1, double lambda2, double shared) {
		return lambda1 >= 0.0 && lambda2 >= 0.0 && shared >= 0.0 &&
				Double.isFinite(lambda1) && Double.isFinite(lambda2) &&
				Double.isFinite(shared);
	}

	public static double density(double x, double y, double lambda1,
			double lambda2, double shared, boolean giveLog) {
		if (!valid(lambda1, lambda2, shared)) return Double.NaN;
		if (x < 0.0 || y < 0.0 || x != Math.rint(x) || y != Math.rint(y) ||
				!Double.isFinite(x) || !Double.isFinite(y))
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		if (shared == 0.0) {
			double logMass = Poisson.density(x, lambda1, true) +
					Poisson.density(y, lambda2, true);
			return giveLog ? logMass : Math.exp(logMass);
		}
		if (Math.min(x, y) > Integer.MAX_VALUE)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		int maximumShared = (int) Math.min(x, y);
		double logMass = Double.NEGATIVE_INFINITY;
		for (int k = 0; k <= maximumShared; k++) {
			double term = Poisson.density(x - k, lambda1, true) +
					Poisson.density(y - k, lambda2, true) +
					Poisson.density(k, shared, true);
			logMass = MultivariateDistributionUtil.logAdd(logMass, term);
		}
		return giveLog ? logMass : Math.exp(logMass);
	}

	/** Returns {@code P[X <= x, Y <= y]}. */
	public static double cumulative(double x, double y, double lambda1,
			double lambda2, double shared, boolean logP) {
		if (!valid(lambda1, lambda2, shared)) return Double.NaN;
		if (Double.isNaN(x) || Double.isNaN(y)) return x + y;
		if (x < 0.0 || y < 0.0) return logP ? Double.NEGATIVE_INFINITY : 0.0;
		if (x == Double.POSITIVE_INFINITY && y == Double.POSITIVE_INFINITY)
			return logP ? 0.0 : 1.0;
		if (x == Double.POSITIVE_INFINITY)
			return Poisson.cumulative(y, lambda2 + shared, true, logP);
		if (y == Double.POSITIVE_INFINITY)
			return Poisson.cumulative(x, lambda1 + shared, true, logP);
		if (shared == 0.0) {
			double logProbability = Poisson.cumulative(x, lambda1, true, true) +
					Poisson.cumulative(y, lambda2, true, true);
			return logP ? logProbability : Math.exp(logProbability);
		}
		int maximumShared = (int) Math.min(Math.floor(x), Math.floor(y));
		double logProbability = Double.NEGATIVE_INFINITY;
		for (int k = 0; k <= maximumShared; k++) {
			double term = Poisson.density(k, shared, true) +
					Poisson.cumulative(x - k, lambda1, true, true) +
					Poisson.cumulative(y - k, lambda2, true, true);
			logProbability = MultivariateDistributionUtil.logAdd(logProbability, term);
		}
		return logP ? logProbability : Math.exp(logProbability);
	}

	public static int[] random(double lambda1, double lambda2, double shared,
			RandomEngine random) {
		if (!valid(lambda1, lambda2, shared) || random == null) return null;
		int common = (int) Poisson.random(shared, random);
		return new int[] {(int) Poisson.random(lambda1, random) + common,
				(int) Poisson.random(lambda2, random) + common};
	}

	public static int[][] random(int n, double lambda1, double lambda2,
			double shared, RandomEngine random) {
		if (n < 0) return null;
		int[][] result = new int[n][];
		for (int i = 0; i < n; i++)
			result[i] = random(lambda1, lambda2, shared, random);
		return result;
	}
}
