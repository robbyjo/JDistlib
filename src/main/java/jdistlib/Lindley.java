/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** One-parameter Lindley lifetime distribution. */
public class Lindley extends GenericDistribution {
	private static boolean invalid(double theta) {
		return !(theta > 0.0) || Double.isInfinite(theta);
	}

	public static double density(double x, double theta, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(theta)) return x + theta;
		if (invalid(theta)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = 2.0 * log(theta) + log1p(x) - theta * x
				- log1p(theta);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double theta, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(theta)) return x + theta;
		if (invalid(theta)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double logUpper = -theta * x + log1p(x / (1.0 + 1.0 / theta));
		double result = lowerTail ? DistributionUtil.logOneMinusExp(logUpper)
				: logUpper;
		return logP ? result : exp(result);
	}

	public static double quantile(double p, double theta, boolean lowerTail,
			boolean logP) {
		if (invalid(theta)) return Double.NaN;
		return DistributionUtil.continuousQuantile(p, lowerTail, logP, 0.0,
				1.0 / theta,
				(x, lt, lp) -> cumulative(x, theta, lt, lp));
	}

	public static double random(double theta, RandomEngine random) {
		if (invalid(theta)) return Double.NaN;
		if (random.nextDouble() < theta / (1.0 + theta)) {
			return Exponential.random(1.0 / theta, random);
		}
		return Gamma.random(2.0, 1.0 / theta, random);
	}
	public static double[] random(int n, double theta, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(theta, random);
		return result;
	}

	private final double theta;
	public Lindley(double theta) { this.theta = theta; }
	@Override public double density(double x, boolean logP) {
		return density(x, theta, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, theta, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, theta, lowerTail, logP);
	}
	@Override public double random() { return random(theta, random); }
}
