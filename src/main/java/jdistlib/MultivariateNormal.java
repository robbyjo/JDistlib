/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Multivariate normal density and random generation using a covariance matrix. */
public final class MultivariateNormal {
	private static final long PROBABILITY_SEED = 0x4a446973746c6962L;

	private MultivariateNormal() {}

	public static double density(double[] x, double[] mean, double[][] covariance,
			boolean giveLog) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(mean, covariance);
		if (factor == null) return Double.NaN;
		double quadratic = MultivariateDistributionUtil.quadratic(x, mean, factor);
		if (Double.isNaN(quadratic)) return Double.NaN;
		double logDensity = -0.5 * (mean.length * Math.log(2.0 * Math.PI) +
				factor.logDeterminant + quadratic);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] mean, double[][] covariance,
			RandomEngine random) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(mean, covariance);
		if (factor == null || random == null) return null;
		return MultivariateDistributionUtil.transform(mean, factor,
				MultivariateDistributionUtil.standardNormal(mean.length, random), 1.0);
	}

	public static double[][] random(int n, double[] mean, double[][] covariance,
			RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(mean, covariance, random);
		return result;
	}

	/** Computes {@code P[lower <= X <= upper]} with numerical error metadata. */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] mean, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return MultivariateProbability.normal(lower, upper, mean, covariance,
				options, random);
	}

	/** Deterministic convenience overload using call-local randomized shifts. */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] mean, double[][] covariance) {
		return probability(lower, upper, mean, covariance,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P[X[i] <= upper[i] for every i]}. */
	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] mean, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (upper == null) return MultivariateProbability.normal(null, null, mean,
				covariance, options, random);
		double[] lower = new double[upper.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		return probability(lower, upper, mean, covariance, options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] mean, double[][] covariance) {
		return cumulative(upper, mean, covariance,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/**
	 * Finds the scalar q satisfying {@code P[X[i] <= q for every i] = p}.
	 * This is an equicoordinate quantile, not a general vector inverse CDF.
	 */
	public static double equicoordinateQuantile(double p, double[] mean,
			double[][] covariance, MultivariateProbabilityOptions options,
			RandomEngine random) {
		return MultivariateQuantile.normal(p, mean, covariance, options, random);
	}

	public static double equicoordinateQuantile(double p, double[] mean,
			double[][] covariance) {
		return equicoordinateQuantile(p, mean, covariance,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Quantile of the Mahalanobis radius containing probability {@code p}. */
	public static double radialQuantile(double p, int dimension,
			boolean lowerTail, boolean logProbability) {
		if (dimension < 1) return Double.NaN;
		return Math.sqrt(ChiSquare.quantile(p, dimension, lowerTail,
				logProbability));
	}
}
