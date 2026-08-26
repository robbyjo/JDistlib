/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Half-Student-t distribution with degrees of freedom and scale. */
public final class HalfT extends GenericDistribution implements SupportedDistribution {
	private static final double LOG_TWO = Math.log(2.0);
	private final double degreesOfFreedom;
	private final double sigma;

	public HalfT(double degreesOfFreedom, double sigma) {
		this.degreesOfFreedom = degreesOfFreedom;
		this.sigma = sigma;
	}

	private static boolean invalid(double df, double sigma) {
		return !(df > 0.0) || !(sigma > 0.0) || !Double.isFinite(df)
				|| !Double.isFinite(sigma);
	}

	public static double density(double x, double df, double sigma, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(df) || Double.isNaN(sigma)) {
			return x + df + sigma;
		}
		if (invalid(df, sigma)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double value = LOG_TWO + T.density(x / sigma, df, true) - Math.log(sigma);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double df, double sigma,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(df) || Double.isNaN(sigma)) {
			return x + df + sigma;
		}
		if (invalid(df, sigma)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double logSurvival = LOG_TWO
				+ T.cumulative(x / sigma, df, false, true);
		double value = lowerTail
				? DistributionUtil.logOneMinusExp(logSurvival) : logSurvival;
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double p, double df, double sigma,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(df) || Double.isNaN(sigma)) {
			return p + df + sigma;
		}
		if (invalid(df, sigma) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double logSurvival = lowerTail
				? (logP ? DistributionUtil.logOneMinusExp(p) : Math.log1p(-p))
				: (logP ? p : Math.log(p));
		return sigma * T.quantile(logSurvival - LOG_TWO, df, false, true);
	}

	public static double random(double df, double sigma, RandomEngine random) {
		if (invalid(df, sigma)) return Double.NaN;
		return Math.abs(sigma * T.random(df, random));
	}

	@Override public double density(double x, boolean log) {
		return density(x, degreesOfFreedom, sigma, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, degreesOfFreedom, sigma, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, degreesOfFreedom, sigma, lowerTail, logP);
	}
	@Override public double random() { return random(degreesOfFreedom, sigma, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
