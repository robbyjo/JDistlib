/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.Bessel;
import jdistlib.rng.RandomEngine;

/** Difference of two independent Poisson variates. */
public final class Skellam extends GenericDistribution implements SupportedDistribution {
	private final double mu1;
	private final double mu2;

	public Skellam(double mu1, double mu2) { this.mu1 = mu1; this.mu2 = mu2; }

	private static boolean invalid(double mu1, double mu2) {
		return mu1 < 0.0 || mu2 < 0.0 || !Double.isFinite(mu1) || !Double.isFinite(mu2);
	}

	public static double density(double x, double mu1, double mu2, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(mu1) || Double.isNaN(mu2)) {
			return x + mu1 + mu2;
		}
		if (invalid(mu1, mu2)) return Double.NaN;
		if (x != Math.rint(x) || Double.isInfinite(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (mu2 == 0.0) return x < 0.0
				? (log ? Double.NEGATIVE_INFINITY : 0.0)
				: Poisson.density(x, mu1, log);
		if (mu1 == 0.0) return x > 0.0
				? (log ? Double.NEGATIVE_INFINITY : 0.0)
				: Poisson.density(-x, mu2, log);
		double argument = 2.0 * Math.sqrt(mu1 * mu2);
		double scaled = Bessel.i(argument, Math.abs(x), true);
		double value = -(mu1 + mu2) + argument
				+ 0.5 * x * Math.log(mu1 / mu2) + Math.log(scaled);
		return log ? value : Math.exp(value);
	}

	private static double mixtureTail(long k, double mu1, double mu2,
			boolean lowerTail) {
		if (mu2 == 0.0) return Poisson.cumulative(k, mu1, lowerTail, true);
		if (mu1 == 0.0) return Poisson.cumulative(-k - 1.0, mu2, !lowerTail, true);
		double mixingMean = lowerTail ? mu2 : mu1;
		double otherMean = lowerTail ? mu1 : mu2;
		double cutoffValue = Poisson.quantile(Math.log(1e-15), mixingMean,
				false, true);
		long cutoff = Math.max(0L, (long) cutoffValue);
		double logSum = Double.NEGATIVE_INFINITY;
		for (long j = 0L; j <= cutoff; j++) {
			double threshold = lowerTail ? j + k : j - k - 1.0;
			double conditional = Poisson.cumulative(threshold, otherMean,
					true, true);
			logSum = DistributionUtil.logAdd(logSum,
					Poisson.density(j, mixingMean, true) + conditional);
		}
		return logSum;
	}

	public static double cumulative(double x, double mu1, double mu2,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mu1) || Double.isNaN(mu2)) {
			return x + mu1 + mu2;
		}
		if (invalid(mu1, mu2)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double value = mixtureTail((long) Math.floor(x), mu1, mu2, lowerTail);
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double p, double mu1, double mu2,
			boolean lowerTail, boolean logP) {
		if (invalid(mu1, mu2) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double target = logP ? Math.exp(p) : p;
		if (!lowerTail) target = 1.0 - target;
		if (target <= 0.0) return Double.NEGATIVE_INFINITY;
		if (target >= 1.0) return Double.POSITIVE_INFINITY;
		long low = (long) Math.floor(mu1 - mu2) - 1L;
		long high = low + 1L;
		long step = 1L;
		while (cumulative(low, mu1, mu2, true, false) >= target) {
			high = low;
			low -= step;
			step = Math.min(1L << 60, step * 2L);
		}
		step = 1L;
		while (cumulative(high, mu1, mu2, true, false) < target) {
			low = high;
			high += step;
			step = Math.min(1L << 60, step * 2L);
		}
		while (high - low > 1L) {
			long middle = low + (high - low) / 2L;
			if (cumulative(middle, mu1, mu2, true, false) >= target) high = middle;
			else low = middle;
		}
		return high;
	}

	public static double random(double mu1, double mu2, RandomEngine random) {
		if (invalid(mu1, mu2)) return Double.NaN;
		return Poisson.random(mu1, random) - Poisson.random(mu2, random);
	}

	@Override public double density(double x, boolean log) { return density(x, mu1, mu2, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu1, mu2, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu1, mu2, lowerTail, logP);
	}
	@Override public double random() { return random(mu1, mu2, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
