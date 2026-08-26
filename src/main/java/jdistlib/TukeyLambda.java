/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Tukey lambda distribution defined by its symmetric quantile function. */
public final class TukeyLambda extends GenericDistribution
		implements SupportedDistribution {
	private final double lambda;

	public TukeyLambda(double lambda) { this.lambda = lambda; }

	public static double quantile(double p, double lambda, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(lambda)) return p + lambda;
		if (!Double.isFinite(lambda) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		if (!lowerTail) probability = 1.0 - probability;
		if (lambda == 0.0) return Math.log(probability) - Math.log1p(-probability);
		return (Math.pow(probability, lambda)
				- Math.pow(1.0 - probability, lambda)) / lambda;
	}

	private static double probability(double x, double lambda) {
		double lower = 0.0;
		double upper = 1.0;
		for (int i = 0; i < 120; i++) {
			double middle = (lower + upper) * 0.5;
			if (quantile(middle, lambda, true, false) >= x) upper = middle;
			else lower = middle;
		}
		return (lower + upper) * 0.5;
	}

	public static double cumulative(double x, double lambda, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(lambda)) return x + lambda;
		if (!Double.isFinite(lambda)) return Double.NaN;
		double bound = lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
		if (x <= -bound) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x >= bound) return DistributionUtil.boundary(true, lowerTail, logP);
		double p = probability(x, lambda);
		double requested = lowerTail ? p : 1.0 - p;
		return logP ? Math.log(requested) : requested;
	}

	public static double density(double x, double lambda, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(lambda)) return x + lambda;
		if (!Double.isFinite(lambda)) return Double.NaN;
		double bound = lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
		if (Math.abs(x) > bound || Double.isInfinite(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (lambda == 1.0) return log ? -Math.log(2.0) : 0.5;
		double p = probability(x, lambda);
		double logDerivative = logSum((lambda - 1.0) * Math.log(p),
				(lambda - 1.0) * Math.log1p(-p));
		return log ? -logDerivative : Math.exp(-logDerivative);
	}

	private static double logSum(double a, double b) {
		double high = Math.max(a, b);
		return high + Math.log(Math.exp(a - high) + Math.exp(b - high));
	}

	public static double random(double lambda, RandomEngine random) {
		if (!Double.isFinite(lambda)) return Double.NaN;
		return quantile(random.nextDouble(), lambda, true, false);
	}

	@Override public double density(double x, boolean log) { return density(x, lambda, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, lambda, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, lambda, lowerTail, logP);
	}
	@Override public double random() { return random(lambda, random); }
	@Override public double getLowerBound() {
		return lambda > 0.0 ? -1.0 / lambda : Double.NEGATIVE_INFINITY;
	}
	@Override public double getUpperBound() {
		return lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
	}
}
