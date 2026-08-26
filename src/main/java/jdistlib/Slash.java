/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Location-scale slash distribution, {@code mu + sigma * Z / U}. */
public final class Slash extends GenericDistribution implements SupportedDistribution {
	private static final double PHI_ZERO = 0.39894228040143267794;
	private final double mu;
	private final double sigma;

	public Slash(double mu, double sigma) { this.mu = mu; this.sigma = sigma; }

	private static boolean invalid(double mu, double sigma) {
		return !Double.isFinite(mu) || !(sigma > 0.0) || !Double.isFinite(sigma);
	}

	public static double density(double x, double mu, double sigma, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma)) {
			return x + mu + sigma;
		}
		if (invalid(mu, sigma)) return Double.NaN;
		if (Double.isInfinite(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double z = (x - mu) / sigma;
		double value;
		if (Math.abs(z) < 1e-4) {
			double z2 = z * z;
			value = PHI_ZERO * (0.5 - z2 / 8.0 + z2 * z2 / 48.0) / sigma;
		} else {
			value = (PHI_ZERO - Normal.density(z, 0.0, 1.0, false))
					/ (z * z * sigma);
		}
		return log ? Math.log(value) : value;
	}

	public static double cumulative(double x, double mu, double sigma,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma)) {
			return x + mu + sigma;
		}
		if (invalid(mu, sigma)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double z = (x - mu) / sigma;
		if (!lowerTail) z = -z;
		double probability;
		if (z == 0.0) probability = 0.5;
		else probability = Normal.cumulative(z, 0.0, 1.0, true, false)
				- (PHI_ZERO - Normal.density(z, 0.0, 1.0, false)) / z;
		return logP ? Math.log(probability) : probability;
	}

	public static double quantile(double p, double mu, double sigma,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(mu) || Double.isNaN(sigma)) {
			return p + mu + sigma;
		}
		if (invalid(mu, sigma) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		if (!lowerTail) probability = 1.0 - probability;
		if (probability == 0.0) return Double.NEGATIVE_INFINITY;
		if (probability == 1.0) return Double.POSITIVE_INFINITY;
		if (probability == 0.5) return mu;
		boolean negative = probability < 0.5;
		double target = negative ? probability : 1.0 - probability;
		double low = 0.0;
		double high = 1.0;
		while (cumulative(mu - high * sigma, mu, sigma, true, false) > target) {
			high *= 2.0;
		}
		for (int i = 0; i < 120; i++) {
			double middle = (low + high) * 0.5;
			if (cumulative(mu - middle * sigma, mu, sigma, true, false) <= target) {
				high = middle;
			} else low = middle;
		}
		return mu + (negative ? -1.0 : 1.0) * (low + high) * 0.5 * sigma;
	}

	public static double random(double mu, double sigma, RandomEngine random) {
		if (invalid(mu, sigma)) return Double.NaN;
		return mu + sigma * Normal.random_standard(random) / random.nextDouble();
	}

	@Override public double density(double x, boolean log) { return density(x, mu, sigma, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, sigma, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu, sigma, lowerTail, logP);
	}
	@Override public double random() { return random(mu, sigma, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
