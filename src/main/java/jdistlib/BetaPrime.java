/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/** Beta-prime (beta of the second kind) distribution. */
public final class BetaPrime extends GenericDistribution
		implements SupportedDistribution {
	private final double shape1;
	private final double shape2;

	public BetaPrime(double shape1, double shape2) {
		this.shape1 = shape1;
		this.shape2 = shape2;
	}

	private static boolean invalid(double a, double b) {
		return !(a > 0.0) || !(b > 0.0) || !Double.isFinite(a) || !Double.isFinite(b);
	}

	public static double density(double x, double a, double b, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (invalid(a, b)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (x == 0.0) {
			if (a < 1.0) return Double.POSITIVE_INFINITY;
			if (a > 1.0) return log ? Double.NEGATIVE_INFINITY : 0.0;
			double value = Math.log(b);
			return log ? value : b;
		}
		double value = (a - 1.0) * Math.log(x)
				- (a + b) * Math.log1p(x) - MathFunctions.lbeta(a, b);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double a, double b,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (invalid(a, b)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		if (x <= 1.0) {
			return Beta.cumulative(x / (1.0 + x), a, b, lowerTail, logP);
		}
		return Beta.cumulative(1.0 / (1.0 + x), b, a, !lowerTail, logP);
	}

	public static double quantile(double p, double a, double b,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || invalid(a, b)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double value = Beta.quantile(p, a, b, lowerTail, logP);
		return value / (1.0 - value);
	}

	public static double random(double a, double b, RandomEngine random) {
		if (invalid(a, b)) return Double.NaN;
		double value = Beta.random(a, b, random);
		return value / (1.0 - value);
	}

	@Override public double density(double x, boolean log) {
		return density(x, shape1, shape2, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, shape1, shape2, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, shape1, shape2, lowerTail, logP);
	}
	@Override public double random() { return random(shape1, shape2, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
