/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.expm1;
import static java.lang.Math.log;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Makeham survival distribution with scale, shape, and constant hazard. */
public class Makeham extends GenericDistribution {
	private static boolean invalid(double scale, double shape, double epsilon) {
		return !(scale > 0.0) || !(shape > 0.0) || epsilon < 0.0
				|| Double.isInfinite(scale) || Double.isInfinite(shape)
				|| Double.isInfinite(epsilon);
	}

	private static double logSurvival(double x, double scale, double shape,
			double epsilon) {
		return -epsilon * x - shape / scale * expm1(scale * x);
	}

	public static double density(double x, double scale, double shape,
			double epsilon, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(shape)
				|| Double.isNaN(epsilon)) return x + scale + shape + epsilon;
		if (invalid(scale, shape, epsilon)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double scaled = scale * x;
		double result = log(epsilon * exp(-scaled) + shape)
				+ x * (scale - epsilon) - shape / scale * expm1(scaled);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double scale, double shape,
			double epsilon, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(shape)
				|| Double.isNaN(epsilon)) return x + scale + shape + epsilon;
		if (invalid(scale, shape, epsilon)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double result = logSurvival(x, scale, shape, epsilon);
		if (lowerTail) result = DistributionUtil.logOneMinusExp(result);
		return logP ? result : exp(result);
	}

	public static double quantile(double p, double scale, double shape,
			double epsilon, boolean lowerTail, boolean logP) {
		if (invalid(scale, shape, epsilon)) return Double.NaN;
		return DistributionUtil.continuousQuantile(p, lowerTail, logP, 0.0,
				1.0 / (shape + epsilon),
				(x, lt, lp) -> cumulative(x, scale, shape, epsilon, lt, lp));
	}

	public static double random(double scale, double shape, double epsilon,
			RandomEngine random) {
		return quantile(random.nextDouble(), scale, shape, epsilon, true, false);
	}
	public static double[] random(int n, double scale, double shape,
			double epsilon, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(scale, shape, epsilon, random);
		return result;
	}

	private final double scale, shape, epsilon;
	public Makeham(double scale, double shape, double epsilon) {
		this.scale = scale; this.shape = shape; this.epsilon = epsilon;
	}
	@Override public double density(double x, boolean logP) {
		return density(x, scale, shape, epsilon, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, scale, shape, epsilon, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, scale, shape, epsilon, lowerTail, logP);
	}
	@Override public double random() { return random(scale, shape, epsilon, random); }
}
