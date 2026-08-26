/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/** Beta-negative-binomial distribution using the extraDistr parameterization. */
public final class BetaNegativeBinomial extends GenericDistribution
		implements SupportedDistribution {
	private final double size;
	private final double alpha;
	private final double beta;

	public BetaNegativeBinomial(double size, double alpha, double beta) {
		this.size = size;
		this.alpha = alpha;
		this.beta = beta;
	}

	private static boolean invalid(double size, double alpha, double beta) {
		return !(size > 0.0) || !(alpha > 0.0) || !(beta > 0.0)
				|| !Double.isFinite(size) || !Double.isFinite(alpha)
				|| !Double.isFinite(beta);
	}

	public static double density(double x, double size, double alpha,
			double beta, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(alpha)
				|| Double.isNaN(beta)) return x + size + alpha + beta;
		if (invalid(size, alpha, beta)) return Double.NaN;
		if (x < 0.0 || x != Math.rint(x) || Double.isInfinite(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double value = MathFunctions.lgammafn(size + x)
				- MathFunctions.lgammafn(x + 1.0) - MathFunctions.lgammafn(size)
				+ MathFunctions.lbeta(alpha + size, beta + x)
				- MathFunctions.lbeta(alpha, beta);
		return log ? value : Math.exp(value);
	}

	private static double lowerProbability(long maximum, double size,
			double alpha, double beta) {
		double term = Math.exp(MathFunctions.lbeta(alpha + size, beta)
				- MathFunctions.lbeta(alpha, beta));
		double sum = term;
		double correction = 0.0;
		for (long k = 0; k < maximum; k++) {
			term *= ((size + k) / (k + 1.0))
					* ((beta + k) / (alpha + beta + size + k));
			double adjusted = term - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
			if (term == 0.0) break;
		}
		return Math.min(1.0, sum);
	}

	public static double cumulative(double x, double size, double alpha,
			double beta, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(alpha)
				|| Double.isNaN(beta)) return x + size + alpha + beta;
		if (invalid(size, alpha, beta)) return Double.NaN;
		if (x < 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		if (x > Long.MAX_VALUE) return DistributionUtil.boundary(true, lowerTail, logP);
		double probability = lowerProbability((long) Math.floor(x), size, alpha, beta);
		double requested = lowerTail ? probability : 1.0 - probability;
		return logP ? Math.log(requested) : requested;
	}

	public static double quantile(double p, double size, double alpha,
			double beta, boolean lowerTail, boolean logP) {
		if (invalid(size, alpha, beta) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double target = logP ? Math.exp(p) : p;
		if (!lowerTail) target = 1.0 - target;
		if (target <= 0.0) return 0.0;
		if (target >= 1.0) return Double.POSITIVE_INFINITY;
		double term = Math.exp(MathFunctions.lbeta(alpha + size, beta)
				- MathFunctions.lbeta(alpha, beta));
		double sum = term;
		long k = 0L;
		while (sum < target && k < Integer.MAX_VALUE) {
			term *= ((size + k) / (k + 1.0))
					* ((beta + k) / (alpha + beta + size + k));
			sum += term;
			k++;
			if (term == 0.0 || sum == 1.0) break;
		}
		return sum >= target ? k : Double.POSITIVE_INFINITY;
	}

	public static double random(double size, double alpha, double beta,
			RandomEngine random) {
		if (invalid(size, alpha, beta)) return Double.NaN;
		double probability = Beta.random(alpha, beta, random);
		return NegBinomial.random(size, probability, random);
	}

	@Override public double density(double x, boolean log) {
		return density(x, size, alpha, beta, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, size, alpha, beta, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, size, alpha, beta, lowerTail, logP);
	}
	@Override public double random() { return random(size, alpha, beta, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
