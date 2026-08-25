/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import java.util.Arrays;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Elliptical multivariate Student t distribution. */
public final class MultivariateStudentT {
	private static final long PROBABILITY_SEED = 0x4a446973746c6962L;

	private MultivariateStudentT() {}

	public static double density(double[] x, double[] location, double[][] scale,
			double degreesOfFreedom, boolean giveLog) {
		if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom))
			return Double.NaN;
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, scale);
		if (factor == null) return Double.NaN;
		double quadratic = MultivariateDistributionUtil.quadratic(x, location, factor);
		if (Double.isNaN(quadratic)) return Double.NaN;
		int dimension = location.length;
		double logDensity = lgammafn((degreesOfFreedom + dimension) / 2.0) -
				lgammafn(degreesOfFreedom / 2.0) -
				0.5 * (dimension * Math.log(degreesOfFreedom * Math.PI) +
				factor.logDeterminant) - 0.5 * (degreesOfFreedom + dimension) *
				Math.log1p(quadratic / degreesOfFreedom);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] location, double[][] scale,
			double degreesOfFreedom, RandomEngine random) {
		if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom)) return null;
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, scale);
		if (factor == null || random == null) return null;
		double chiSquare = ChiSquare.random(degreesOfFreedom, random);
		return MultivariateDistributionUtil.transform(location, factor,
				MultivariateDistributionUtil.standardNormal(location.length, random),
				Math.sqrt(degreesOfFreedom / chiSquare));
	}

	public static double[][] random(int n, double[] location, double[][] scale,
			double degreesOfFreedom, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++)
			result[i] = random(location, scale, degreesOfFreedom, random);
		return result;
	}

	/** Computes {@code P[lower <= X <= upper]} with numerical error metadata. */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] scale,
			double degreesOfFreedom, MultivariateProbabilityOptions options,
			RandomEngine random) {
		return MultivariateProbability.studentT(lower, upper, location, scale,
				degreesOfFreedom, options, random);
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] scale,
			double degreesOfFreedom) {
		return probability(lower, upper, location, scale, degreesOfFreedom,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P[X[i] <= upper[i] for every i]}. */
	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scale, double degreesOfFreedom,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (upper == null) return MultivariateProbability.studentT(null, null,
				location, scale, degreesOfFreedom, options, random);
		double[] lower = new double[upper.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		return probability(lower, upper, location, scale, degreesOfFreedom,
				options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scale, double degreesOfFreedom) {
		return cumulative(upper, location, scale, degreesOfFreedom,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Equicoordinate quantile, with one common threshold in every dimension. */
	public static double equicoordinateQuantile(double p, double[] location,
			double[][] scale, double degreesOfFreedom,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return MultivariateQuantile.studentT(p, location, scale, degreesOfFreedom,
				options, random);
	}

	public static double equicoordinateQuantile(double p, double[] location,
			double[][] scale, double degreesOfFreedom) {
		return equicoordinateQuantile(p, location, scale, degreesOfFreedom,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Quantile of the Mahalanobis radius containing probability {@code p}. */
	public static double radialQuantile(double p, int dimension,
			double degreesOfFreedom, boolean lowerTail, boolean logProbability) {
		if (dimension < 1 || !(degreesOfFreedom > 0.0) ||
				!Double.isFinite(degreesOfFreedom)) return Double.NaN;
		return Math.sqrt(dimension * F.quantile(p, dimension, degreesOfFreedom,
				lowerTail, logProbability));
	}
}
