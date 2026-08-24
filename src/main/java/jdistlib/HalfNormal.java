/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Half-normal distribution, the distribution of the absolute value of a
 * zero-centered normal variate with scale {@code sigma}.
 *
 * <p>The parameterization follows the GPL-2
 * <a href="https://cran.r-project.org/package=extraDistr">extraDistr</a>
 * {@code dhnorm} API.</p>
 *
 * @author Roby Joehanes
 */
public class HalfNormal extends GenericDistribution {
	private static final double LOG_2 = log(2.0);

	private static double logOneMinusExp(double x) {
		return x > -LOG_2 ? log(-Math.expm1(x)) : log1p(-exp(x));
	}

	public static double density(double x, double sigma, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (!(sigma > 0.0) || Double.isInfinite(sigma)) return Double.NaN;
		if (x < 0.0) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		double result = LOG_2 + Normal.density(x, 0.0, sigma, true);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(sigma)) return x + sigma;
		if (!(sigma > 0.0) || Double.isInfinite(sigma)) return Double.NaN;
		if (x <= 0.0) return boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) return boundary(true, lowerTail, logP);

		double logSurvival = LOG_2
				+ Normal.cumulative(x, 0.0, sigma, false, true);
		double result = lowerTail ? logOneMinusExp(logSurvival) : logSurvival;
		return logP ? result : exp(result);
	}

	public static double quantile(double p, double sigma, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(sigma)) return p + sigma;
		if (!(sigma > 0.0) || Double.isInfinite(sigma)
				|| invalidProbability(p, logP)) return Double.NaN;

		double logSurvival;
		if (lowerTail) {
			logSurvival = logP ? logOneMinusExp(p) : log1p(-p);
		} else {
			logSurvival = logP ? p : log(p);
		}
		return Normal.quantile(logSurvival - LOG_2, 0.0, sigma, false, true);
	}

	public static double random(double sigma, RandomEngine random) {
		if (!(sigma > 0.0) || Double.isInfinite(sigma)) return Double.NaN;
		return abs(sigma * Normal.random_standard(random));
	}

	public static double[] random(int n, double sigma, RandomEngine random) {
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = random(sigma, random);
		return values;
	}

	private static boolean invalidProbability(double p, boolean logP) {
		return logP ? p > 0.0 : p < 0.0 || p > 1.0;
	}

	private static double boundary(boolean upper, boolean lowerTail, boolean logP) {
		boolean one = upper == lowerTail;
		return logP ? (one ? 0.0 : Double.NEGATIVE_INFINITY) : (one ? 1.0 : 0.0);
	}

	private final double sigma;

	public HalfNormal(double sigma) {
		this.sigma = sigma;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, sigma, log);
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
