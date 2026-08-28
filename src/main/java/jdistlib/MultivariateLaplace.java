/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import java.util.Arrays;

import jdistlib.math.Bessel;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Symmetric multivariate Laplace law defined as a normal-exponential mixture. */
public final class MultivariateLaplace {
	private static final long PROBABILITY_SEED = 0x4d764c61706c6163L;
	private MultivariateLaplace() {}

	public static double density(double[] x, double[] location,
			double[][] covariance, boolean giveLog) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, covariance);
		if (factor == null) return Double.NaN;
		double quadratic = MultivariateDistributionUtil.quadratic(x, location, factor);
		if (Double.isNaN(quadratic)) return Double.NaN;
		if (Double.isInfinite(quadratic))
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		int dimension = location.length;
		double logDensity;
		if (quadratic == 0.0) {
			if (dimension >= 2) logDensity = Double.POSITIVE_INFINITY;
			else logDensity = lgammafn(0.5) - 0.5 * Math.log(2.0 * Math.PI) -
					0.5 * factor.logDeterminant;
		} else {
			double argument = Math.sqrt(2.0 * quadratic);
			double order = 1.0 - dimension / 2.0;
			double scaledBessel = Bessel.k(argument, order, true);
			logDensity = Math.log(2.0) - 0.5 * dimension * Math.log(2.0 * Math.PI) -
					0.5 * factor.logDeterminant + 0.5 * order *
					Math.log(quadratic / 2.0) + Math.log(scaledBessel) - argument;
		}
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] location, double[][] covariance,
			RandomEngine random) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, covariance);
		if (factor == null || random == null) return null;
		double mixing = Exponential.random(1.0, random);
		return MultivariateDistributionUtil.transform(location, factor,
				MultivariateDistributionUtil.standardNormal(location.length, random),
				Math.sqrt(mixing));
	}

	public static double[][] random(int n, double[] location,
			double[][] covariance, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(location, covariance, random);
		return result;
	}

	/** Computes {@code P(lower <= X <= upper)} through the normal-exponential mixture. */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return MultivariateProbability.laplace(lower, upper, location, covariance,
				options, random);
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] covariance) {
		return probability(lower, upper, location, covariance,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P(X[i] <= upper[i], all i)}. */
	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (upper == null) return MultivariateProbability.laplace(null, null,
				location, covariance, options, random);
		double[] lower = new double[upper.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		return probability(lower, upper, location, covariance, options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] covariance) {
		return cumulative(upper, location, covariance,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}
}
