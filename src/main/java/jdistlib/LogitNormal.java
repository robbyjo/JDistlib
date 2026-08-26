/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Logit-normal distribution on the open unit interval. */
public final class LogitNormal extends GenericDistribution
		implements SupportedDistribution {
	private final double meanLogit;
	private final double standardDeviationLogit;

	public LogitNormal(double meanLogit, double standardDeviationLogit) {
		this.meanLogit = meanLogit;
		this.standardDeviationLogit = standardDeviationLogit;
	}

	private static boolean invalid(double mean, double standardDeviation) {
		return !Double.isFinite(mean) || !(standardDeviation > 0.0)
				|| !Double.isFinite(standardDeviation);
	}

	private static double logistic(double x) {
		return x >= 0.0 ? 1.0 / (1.0 + Math.exp(-x))
				: Math.exp(x) / (1.0 + Math.exp(x));
	}

	public static double density(double x, double mean, double standardDeviation,
			boolean log) {
		if (Double.isNaN(x) || Double.isNaN(mean)
				|| Double.isNaN(standardDeviation)) return x + mean + standardDeviation;
		if (invalid(mean, standardDeviation)) return Double.NaN;
		if (!(x > 0.0 && x < 1.0)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double logit = Math.log(x) - Math.log1p(-x);
		double value = Normal.density(logit, mean, standardDeviation, true)
				- Math.log(x) - Math.log1p(-x);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double mean, double standardDeviation,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mean)
				|| Double.isNaN(standardDeviation)) return x + mean + standardDeviation;
		if (invalid(mean, standardDeviation)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x >= 1.0) return DistributionUtil.boundary(true, lowerTail, logP);
		return Normal.cumulative(Math.log(x) - Math.log1p(-x), mean,
				standardDeviation, lowerTail, logP);
	}

	public static double quantile(double p, double mean, double standardDeviation,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || invalid(mean, standardDeviation)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		return logistic(Normal.quantile(p, mean, standardDeviation, lowerTail, logP));
	}

	public static double random(double mean, double standardDeviation,
			RandomEngine random) {
		if (invalid(mean, standardDeviation)) return Double.NaN;
		return logistic(mean + standardDeviation * Normal.random_standard(random));
	}

	@Override public double density(double x, boolean log) {
		return density(x, meanLogit, standardDeviationLogit, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, meanLogit, standardDeviationLogit, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, meanLogit, standardDeviationLogit, lowerTail, logP);
	}
	@Override public double random() {
		return random(meanLogit, standardDeviationLogit, random);
	}
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return 1.0; }
}
