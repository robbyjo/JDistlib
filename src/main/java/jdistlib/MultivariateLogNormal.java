/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Component-wise exponential transform of a multivariate normal vector. */
public final class MultivariateLogNormal {
	private MultivariateLogNormal() {}

	public static double density(double[] x, double[] meanLog,
			double[][] covarianceLog, boolean giveLog) {
		if (x == null) return Double.NaN;
		double[] logged = new double[x.length];
		double jacobian = 0.0;
		for (int i = 0; i < x.length; i++) {
			if (!(x[i] > 0.0) || !Double.isFinite(x[i]))
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			logged[i] = Math.log(x[i]);
			jacobian += logged[i];
		}
		double logDensity = MultivariateNormal.density(logged, meanLog,
				covarianceLog, true) - jacobian;
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] meanLog, double[][] covarianceLog,
			RandomEngine random) {
		double[] result = MultivariateNormal.random(meanLog, covarianceLog, random);
		if (result == null) return null;
		for (int i = 0; i < result.length; i++) result[i] = Math.exp(result[i]);
		return result;
	}

	public static double[][] random(int n, double[] meanLog,
			double[][] covarianceLog, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(meanLog, covarianceLog, random);
		return result;
	}

	/** Computes a rectangular probability after the component-wise log transform. */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] meanLog, double[][] covarianceLog,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (lower == null || upper == null || lower.length != upper.length)
			return MultivariateNormal.probability(null, null, meanLog, covarianceLog,
					options, random);
		double[] loggedLower = new double[lower.length];
		double[] loggedUpper = new double[upper.length];
		for (int i = 0; i < lower.length; i++) {
			if (Double.isNaN(lower[i]) || Double.isNaN(upper[i]))
				return MultivariateNormal.probability(null, null, meanLog,
						covarianceLog, options, random);
			loggedLower[i] = lower[i] <= 0.0 ? Double.NEGATIVE_INFINITY :
					Math.log(lower[i]);
			if (upper[i] <= 0.0) loggedUpper[i] = Double.NEGATIVE_INFINITY;
			else loggedUpper[i] = Math.log(upper[i]);
		}
		return MultivariateNormal.probability(loggedLower, loggedUpper, meanLog,
				covarianceLog, options, random);
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] meanLog, double[][] covarianceLog) {
		return probability(lower, upper, meanLog, covarianceLog,
				new MultivariateProbabilityOptions(),
				new jdistlib.rng.MersenneTwister(0x4a446973746c6962L));
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] meanLog, double[][] covarianceLog,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (upper == null) return probability(null, null, meanLog, covarianceLog,
				options, random);
		double[] lower = new double[upper.length];
		Arrays.fill(lower, 0.0);
		return probability(lower, upper, meanLog, covarianceLog, options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] meanLog, double[][] covarianceLog) {
		if (upper == null) return probability(null, null, meanLog, covarianceLog);
		double[] lower = new double[upper.length];
		Arrays.fill(lower, 0.0);
		return probability(lower, upper, meanLog, covarianceLog);
	}

	/** Equicoordinate quantile on the original, positive measurement scale. */
	public static double equicoordinateQuantile(double p, double[] meanLog,
			double[][] covarianceLog, MultivariateProbabilityOptions options,
			RandomEngine random) {
		double logQuantile = MultivariateNormal.equicoordinateQuantile(p, meanLog,
				covarianceLog, options, random);
		return Math.exp(logQuantile);
	}

	public static double equicoordinateQuantile(double p, double[] meanLog,
			double[][] covarianceLog) {
		return Math.exp(MultivariateNormal.equicoordinateQuantile(p, meanLog,
				covarianceLog));
	}
}
