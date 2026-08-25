/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * VGAM folded normal distribution, including asymmetric positive and negative
 * scaling factors {@code a1} and {@code a2}.
 */
public class FoldedNormal extends GenericDistribution {
	private static boolean invalid(double mean, double sd, double a1, double a2) {
		return Double.isNaN(mean) || Double.isInfinite(mean) || !(sd > 0.0)
				|| Double.isInfinite(sd) || !(a1 > 0.0) || Double.isInfinite(a1)
				|| !(a2 > 0.0) || Double.isInfinite(a2);
	}

	public static double density(double x, double mean, double sd, double a1,
			double a2, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(sd)
				|| Double.isNaN(a1) || Double.isNaN(a2)) return x + mean + sd + a1 + a2;
		if (invalid(mean, sd, a1, a2)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double first = Normal.density(x / a1, mean, sd, false) / a1;
		double second = Normal.density(-x / a2, mean, sd, false) / a2;
		double result = first + second;
		return giveLog ? Math.log(result) : result;
	}

	public static double cumulative(double x, double mean, double sd, double a1,
			double a2, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(sd)
				|| Double.isNaN(a1) || Double.isNaN(a2)) return x + mean + sd + a1 + a2;
		if (invalid(mean, sd, a1, a2)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double result;
		if (lowerTail) {
			result = Normal.cumulative(x / a1, mean, sd, true, false)
					- Normal.cumulative(-x / a2, mean, sd, true, false);
		} else {
			result = Normal.cumulative(x / a1, mean, sd, false, false)
					+ Normal.cumulative(-x / a2, mean, sd, true, false);
		}
		return logP ? Math.log(result) : result;
	}

	public static double quantile(double p, double mean, double sd, double a1,
			double a2, boolean lowerTail, boolean logP) {
		if (invalid(mean, sd, a1, a2)) return Double.NaN;
		double initial = Math.max(a1, a2) * (Math.abs(mean) + 2.0 * sd);
		return DistributionUtil.continuousQuantile(p, lowerTail, logP, 0.0,
				initial, (x, lt, lp) -> cumulative(x, mean, sd, a1, a2, lt, lp));
	}

	public static double random(double mean, double sd, double a1, double a2,
			RandomEngine random) {
		if (invalid(mean, sd, a1, a2)) return Double.NaN;
		double value = Normal.random(mean, sd, random);
		return Math.max(a1 * value, -a2 * value);
	}
	public static double[] random(int n, double mean, double sd, double a1,
			double a2, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mean, sd, a1, a2, random);
		return result;
	}

	private final double mean, sd, a1, a2;
	public FoldedNormal(double mean, double sd) { this(mean, sd, 1.0, 1.0); }
	public FoldedNormal(double mean, double sd, double a1, double a2) {
		this.mean = mean; this.sd = sd; this.a1 = a1; this.a2 = a2;
	}
	@Override public double density(double x, boolean logP) {
		return density(x, mean, sd, a1, a2, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mean, sd, a1, a2, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mean, sd, a1, a2, lowerTail, logP);
	}
	@Override public double random() { return random(mean, sd, a1, a2, random); }
}
