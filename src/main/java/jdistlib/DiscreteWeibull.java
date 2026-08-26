/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Nakagawa-Osaki type-I discrete Weibull distribution on nonnegative integers. */
public final class DiscreteWeibull extends GenericDistribution
		implements SupportedDistribution {
	private final double q;
	private final double beta;

	public DiscreteWeibull(double q, double beta) { this.q = q; this.beta = beta; }

	private static boolean invalid(double q, double beta) {
		return !(q > 0.0 && q < 1.0) || !(beta > 0.0)
				|| !Double.isFinite(beta);
	}

	private static double logSurvival(double x, double q, double beta) {
		return Math.log(q) * Math.exp(beta * Math.log(x + 1.0));
	}

	public static double density(double x, double q, double beta, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(q) || Double.isNaN(beta)) {
			return x + q + beta;
		}
		if (invalid(q, beta)) return Double.NaN;
		if (x < 0.0 || x != Math.rint(x) || Double.isInfinite(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double a = x == 0.0 ? 0.0
				: Math.log(q) * Math.exp(beta * Math.log(x));
		double b = logSurvival(x, q, beta);
		double value = a + DistributionUtil.logOneMinusExp(b - a);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double q, double beta,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(q) || Double.isNaN(beta)) {
			return x + q + beta;
		}
		if (invalid(q, beta)) return Double.NaN;
		if (x < 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double logSurvival = logSurvival(Math.floor(x), q, beta);
		double value = lowerTail
				? DistributionUtil.logOneMinusExp(logSurvival) : logSurvival;
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double p, double q, double beta,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(q) || Double.isNaN(beta)) {
			return p + q + beta;
		}
		if (invalid(q, beta) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double logS = lowerTail
				? (logP ? DistributionUtil.logOneMinusExp(p) : Math.log1p(-p))
				: (logP ? p : Math.log(p));
		if (logS == 0.0) return 0.0;
		if (logS == Double.NEGATIVE_INFINITY) return Double.POSITIVE_INFINITY;
		double value = Math.ceil(Math.pow(logS / Math.log(q), 1.0 / beta) - 1.0);
		value = Math.max(0.0, value);
		double requested = logP ? Math.exp(p) : p;
		double target = lowerTail ? requested : 1.0 - requested;
		while (value > 0.0
				&& cumulative(value - 1.0, q, beta, true, false) >= target) value--;
		while (cumulative(value, q, beta, true, false) < target) value++;
		return value;
	}

	public static double random(double q, double beta, RandomEngine random) {
		if (invalid(q, beta)) return Double.NaN;
		return quantile(random.nextDouble(), q, beta, true, false);
	}

	@Override public double density(double x, boolean log) { return density(x, q, beta, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, q, beta, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, q, beta, lowerTail, logP);
	}
	@Override public double random() { return random(q, beta, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
