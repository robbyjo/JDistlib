/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Maxwell-Boltzmann speed distribution using the conventional scale
 * {@code sigma}, the common standard deviation of three independent centered
 * normal coordinates.
 *
 * <p>This is the same probability law as {@link Maxwell}, with
 * {@code rate = 1 / (sigma * sigma)}.</p>
 */
public class MaxwellBoltzmann extends GenericDistribution {
	private static boolean invalid(double sigma) {
		return !(sigma > 0.0) || Double.isInfinite(sigma);
	}

	public static double density(double x, double sigma, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (invalid(sigma)) return Double.NaN;
		if (x <= 0.0 || x == Double.POSITIVE_INFINITY)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		double standardized = x / sigma;
		double logDensity = 0.5 * log(2.0 / Math.PI) +
				2.0 * log(standardized) - log(sigma) -
				0.5 * standardized * standardized;
		return giveLog ? logDensity : exp(logDensity);
	}

	public static double cumulative(double x, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (invalid(sigma)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		double standardized = x / sigma;
		return Gamma.cumulative(0.5 * standardized * standardized, 1.5, 1.0,
				lowerTail, logP);
	}

	public static double quantile(double p, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(sigma)) return p + sigma;
		if (invalid(sigma)) return Double.NaN;
		return sigma * sqrt(2.0 * Gamma.quantile(p, 1.5, 1.0,
				lowerTail, logP));
	}

	public static double random(double sigma, RandomEngine random) {
		if (invalid(sigma)) return Double.NaN;
		return sigma * sqrt(2.0 * Gamma.random(1.5, 1.0, random));
	}

	public static double[] random(int n, double sigma, RandomEngine random) {
		if (n < 0) return null;
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(sigma, random);
		return result;
	}

	private final double sigma;

	public MaxwellBoltzmann(double sigma) {
		this.sigma = sigma;
	}

	@Override
	public double density(double x, boolean logP) {
		return density(x, sigma, logP);
	}

	@Override
	public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, sigma, lowerTail, logP);
	}

	@Override
	public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, sigma, lowerTail, logP);
	}

	@Override
	public double random() {
		return random(sigma, random);
	}
}
