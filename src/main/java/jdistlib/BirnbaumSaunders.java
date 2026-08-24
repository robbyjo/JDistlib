/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Birnbaum-Saunders (fatigue-life) distribution with shape {@code alpha},
 * scale {@code beta}, and location {@code mu}.
 *
 * <p>The support is {@code x > mu}. The parameterization follows the GPL-2
 * <a href="https://cran.r-project.org/package=extraDistr">extraDistr</a>
 * {@code dfatigue} API.</p>
 *
 * @author Roby Joehanes
 */
public class BirnbaumSaunders extends GenericDistribution {
	private static boolean invalid(double alpha, double beta, double mu) {
		return !(alpha > 0.0) || !(beta > 0.0)
				|| Double.isInfinite(alpha) || Double.isInfinite(beta)
				|| Double.isNaN(mu) || Double.isInfinite(mu);
	}

	private static double normalArgument(double x, double alpha, double beta,
			double mu) {
		double shifted = x - mu;
		return (sqrt(shifted / beta) - sqrt(beta / shifted)) / alpha;
	}

	private static double transformNormal(double z, double alpha, double beta,
			double mu) {
		double t = 0.5 * alpha * z;
		double root = Math.hypot(t, 1.0);
		double factor;
		if (t >= 0.0) {
			factor = t + root;
		} else {
			factor = 1.0 / (root - t);
		}
		return mu + beta * factor * factor;
	}

	public static double density(double x, double alpha, double beta, double mu,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(alpha) || Double.isNaN(beta)
				|| Double.isNaN(mu)) return x + alpha + beta + mu;
		if (invalid(alpha, beta, mu)) return Double.NaN;
		if (x <= mu || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}

		double shifted = x - mu;
		double rootForward = sqrt(shifted / beta);
		double rootInverse = sqrt(beta / shifted);
		double z = (rootForward - rootInverse) / alpha;
		double result = log(rootForward + rootInverse) - log(2.0) - log(alpha)
				- log(shifted) + Normal.density(z, 0.0, 1.0, true);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double alpha, double beta, double mu,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(alpha) || Double.isNaN(beta)
				|| Double.isNaN(mu)) return x + alpha + beta + mu;
		if (invalid(alpha, beta, mu)) return Double.NaN;
		if (x <= mu) return boundary(false, lowerTail, logP);
		return Normal.cumulative(normalArgument(x, alpha, beta, mu), 0.0, 1.0,
				lowerTail, logP);
	}

	public static double quantile(double p, double alpha, double beta, double mu,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(alpha) || Double.isNaN(beta)
				|| Double.isNaN(mu)) return p + alpha + beta + mu;
		if (invalid(alpha, beta, mu) || invalidProbability(p, logP)) return Double.NaN;
		double z = Normal.quantile(p, 0.0, 1.0, lowerTail, logP);
		return transformNormal(z, alpha, beta, mu);
	}

	public static double random(double alpha, double beta, double mu,
			RandomEngine random) {
		if (invalid(alpha, beta, mu)) return Double.NaN;
		return transformNormal(Normal.random_standard(random), alpha, beta, mu);
	}

	public static double[] random(int n, double alpha, double beta, double mu,
			RandomEngine random) {
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = random(alpha, beta, mu, random);
		return values;
	}

	private static boolean invalidProbability(double p, boolean logP) {
		return logP ? p > 0.0 : p < 0.0 || p > 1.0;
	}

	private static double boundary(boolean upper, boolean lowerTail, boolean logP) {
		boolean one = upper == lowerTail;
		return logP ? (one ? 0.0 : Double.NEGATIVE_INFINITY) : (one ? 1.0 : 0.0);
	}

	private final double alpha;
	private final double beta;
	private final double mu;

	public BirnbaumSaunders(double alpha, double beta, double mu) {
		this.alpha = alpha;
		this.beta = beta;
		this.mu = mu;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, alpha, beta, mu, log);
	}

	@Override
	public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, alpha, beta, mu, lowerTail, logP);
	}

	@Override
	public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, alpha, beta, mu, lowerTail, logP);
	}

	@Override
	public double random() {
		return random(alpha, beta, mu, random);
	}
}
