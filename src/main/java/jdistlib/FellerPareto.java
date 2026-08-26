/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/** Five-parameter Feller-Pareto distribution from actuar. */
public final class FellerPareto extends GenericDistribution
		implements SupportedDistribution {
	private final double minimum;
	private final double shape1;
	private final double shape2;
	private final double shape3;
	private final double scale;

	public FellerPareto(double minimum, double shape1, double shape2,
			double shape3, double scale) {
		this.minimum = minimum;
		this.shape1 = shape1;
		this.shape2 = shape2;
		this.shape3 = shape3;
		this.scale = scale;
	}

	private static boolean invalid(double minimum, double shape1, double shape2,
			double shape3, double scale) {
		return !Double.isFinite(minimum) || !(shape1 > 0.0) || !(shape2 > 0.0)
				|| !(shape3 > 0.0) || !(scale > 0.0) || !Double.isFinite(shape1)
				|| !Double.isFinite(shape2) || !Double.isFinite(shape3)
				|| !Double.isFinite(scale);
	}

	public static double density(double x, double minimum, double shape1,
			double shape2, double shape3, double scale, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(minimum) || Double.isNaN(shape1)
				|| Double.isNaN(shape2) || Double.isNaN(shape3)
				|| Double.isNaN(scale)) {
			return x + minimum + shape1 + shape2 + shape3 + scale;
		}
		if (invalid(minimum, shape1, shape2, shape3, scale)) return Double.NaN;
		if (x < minimum || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (x == minimum) {
			double exponent = shape2 * shape3;
			if (exponent < 1.0) return Double.POSITIVE_INFINITY;
			if (exponent > 1.0) return log ? Double.NEGATIVE_INFINITY : 0.0;
			double value = Math.log(shape2 / scale)
					- MathFunctions.lbeta(shape3, shape1);
			return log ? value : Math.exp(value);
		}
		double logV = shape2 * (Math.log(x - minimum) - Math.log(scale));
		double logU = -logOnePlusExp(-logV);
		double logOneMinusU = -logOnePlusExp(logV);
		double value = Math.log(shape2) + shape3 * logU
				+ shape1 * logOneMinusU - Math.log(x - minimum)
				- MathFunctions.lbeta(shape3, shape1);
		return log ? value : Math.exp(value);
	}

	private static double logOnePlusExp(double x) {
		return x > 0.0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
	}

	public static double cumulative(double x, double minimum, double shape1,
			double shape2, double shape3, double scale, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(minimum) || Double.isNaN(shape1)
				|| Double.isNaN(shape2) || Double.isNaN(shape3)
				|| Double.isNaN(scale)) {
			return x + minimum + shape1 + shape2 + shape3 + scale;
		}
		if (invalid(minimum, shape1, shape2, shape3, scale)) return Double.NaN;
		if (x <= minimum) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double logV = shape2 * (Math.log(x - minimum) - Math.log(scale));
		double u = logV >= 0.0 ? 1.0 / (1.0 + Math.exp(-logV))
				: Math.exp(logV) / (1.0 + Math.exp(logV));
		return Beta.cumulative(u, shape3, shape1, lowerTail, logP);
	}

	public static double quantile(double p, double minimum, double shape1,
			double shape2, double shape3, double scale, boolean lowerTail,
			boolean logP) {
		if (invalid(minimum, shape1, shape2, shape3, scale)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double u = Beta.quantile(p, shape3, shape1, lowerTail, logP);
		if (u == 0.0) return minimum;
		if (u == 1.0) return Double.POSITIVE_INFINITY;
		return minimum + scale * Math.exp((Math.log(u) - Math.log1p(-u)) / shape2);
	}

	public static double random(double minimum, double shape1, double shape2,
			double shape3, double scale, RandomEngine random) {
		if (invalid(minimum, shape1, shape2, shape3, scale)) return Double.NaN;
		double value = Beta.random(shape1, shape3, random);
		return minimum + scale * Math.pow(1.0 / value - 1.0, 1.0 / shape2);
	}

	@Override public double density(double x, boolean log) {
		return density(x, minimum, shape1, shape2, shape3, scale, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, minimum, shape1, shape2, shape3, scale, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, minimum, shape1, shape2, shape3, scale, lowerTail, logP);
	}
	@Override public double random() {
		return random(minimum, shape1, shape2, shape3, scale, random);
	}
	@Override public double getLowerBound() { return minimum; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
