/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Normal distribution left-truncated at zero. */
public class PositiveNormal extends GenericDistribution {
	private static boolean invalid(double mean, double sd) {
		return Double.isNaN(mean) || Double.isInfinite(mean) || !(sd > 0.0)
				|| Double.isInfinite(sd);
	}

	public static double density(double x, double mean, double sd,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(sd)) {
			return x + mean + sd;
		}
		if (invalid(mean, sd)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = Normal.density(x, mean, sd, true)
				- Normal.cumulative(0.0, mean, sd, false, true);
		return giveLog ? result : Math.exp(result);
	}

	public static double cumulative(double x, double mean, double sd,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(sd)) {
			return x + mean + sd;
		}
		if (invalid(mean, sd)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		double logUpper = Normal.cumulative(x, mean, sd, false, true)
				- Normal.cumulative(0.0, mean, sd, false, true);
		double result = lowerTail ? DistributionUtil.logOneMinusExp(logUpper)
				: logUpper;
		return logP ? result : Math.exp(result);
	}

	public static double quantile(double p, double mean, double sd,
			boolean lowerTail, boolean logP) {
		if (invalid(mean, sd)) return Double.NaN;
		return DistributionUtil.continuousQuantile(p, lowerTail, logP, 0.0,
				Math.max(sd, mean + 2.0 * sd),
				(x, lt, lp) -> cumulative(x, mean, sd, lt, lp));
	}

	public static double random(double mean, double sd, RandomEngine random) {
		return quantile(random.nextDouble(), mean, sd, true, false);
	}
	public static double[] random(int n, double mean, double sd,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mean, sd, random);
		return result;
	}

	private final double mean, sd;
	public PositiveNormal(double mean, double sd) { this.mean = mean; this.sd = sd; }
	@Override public double density(double x, boolean logP) {
		return density(x, mean, sd, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mean, sd, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mean, sd, lowerTail, logP);
	}
	@Override public double random() { return random(mean, sd, random); }
}
