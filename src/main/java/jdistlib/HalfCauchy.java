/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Half-Cauchy distribution with positive scale {@code sigma}. */
public final class HalfCauchy extends GenericDistribution
		implements SupportedDistribution {
	private static final double LOG_TWO = Math.log(2.0);
	private final double sigma;

	public HalfCauchy(double sigma) { this.sigma = sigma; }

	public static double density(double x, double sigma, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (!(sigma > 0.0) || !Double.isFinite(sigma)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double value = LOG_TWO - Math.log(Math.PI) - Math.log(sigma)
				- Math.log1p((x / sigma) * (x / sigma));
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (!(sigma > 0.0) || !Double.isFinite(sigma)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double probability = lowerTail
				? 2.0 * Math.atan(x / sigma) / Math.PI
				: 2.0 * Math.atan(sigma / x) / Math.PI;
		return logP ? Math.log(probability) : probability;
	}

	public static double quantile(double p, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(sigma)) return p + sigma;
		if (!(sigma > 0.0) || !Double.isFinite(sigma)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double probability = logP ? Math.exp(p) : p;
		if (probability == 0.0) return lowerTail ? 0.0 : Double.POSITIVE_INFINITY;
		if (probability == 1.0) return lowerTail ? Double.POSITIVE_INFINITY : 0.0;
		return sigma * (lowerTail
				? Math.tan(Math.PI * probability / 2.0)
				: 1.0 / Math.tan(Math.PI * probability / 2.0));
	}

	public static double random(double sigma, RandomEngine random) {
		if (!(sigma > 0.0) || !Double.isFinite(sigma)) return Double.NaN;
		return Math.abs(Cauchy.random(0.0, sigma, random));
	}

	@Override public double density(double x, boolean log) {
		return density(x, sigma, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, sigma, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, sigma, lowerTail, logP);
	}
	@Override public double random() { return random(sigma, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
