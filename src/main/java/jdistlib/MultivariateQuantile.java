/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Shared inversion of equicoordinate multivariate rectangle probabilities. */
final class MultivariateQuantile {
	private MultivariateQuantile() {}

	static double normal(double p, double[] mean, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return invert(p, mean, covariance, 0.0, options, random);
	}

	static double studentT(double p, double[] location, double[][] scale,
			double degreesOfFreedom, MultivariateProbabilityOptions options,
			RandomEngine random) {
		if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom))
			return Double.NaN;
		return invert(p, location, scale, degreesOfFreedom, options, random);
	}

	private static double invert(double p, double[] location, double[][] scale,
			double degreesOfFreedom, MultivariateProbabilityOptions options,
			RandomEngine random) {
		if (Double.isNaN(p) || p < 0.0 || p > 1.0 || location == null ||
				scale == null || options == null || !options.isValid() || random == null ||
				MultivariateDistributionUtil.factor(location, scale) == null)
			return Double.NaN;
		if (p == 0.0) return Double.NEGATIVE_INFINITY;
		if (p == 1.0) return Double.POSITIVE_INFINITY;
		int dimension = location.length;
		double minimumLocation = location[0];
		double maximumLocation = location[0];
		double maximumScale = 0.0;
		for (int i = 0; i < dimension; i++) {
			minimumLocation = Math.min(minimumLocation, location[i]);
			maximumLocation = Math.max(maximumLocation, location[i]);
			maximumScale = Math.max(maximumScale, Math.sqrt(scale[i][i]));
		}
		double marginalTail = Math.max(1e-10, Math.min(0.01,
				Math.min(p, 1.0 - p) / Math.max(1, dimension)));
		double standardized = degreesOfFreedom > 0.0
				? Math.abs(T.quantile(marginalTail, degreesOfFreedom, true, false))
				: Math.abs(Normal.quantile(marginalTail, 0.0, 1.0, true, false));
		if (!Double.isFinite(standardized)) standardized = 8.0;
		double width = Math.max(1.0, standardized * maximumScale);
		double lower = minimumLocation - width;
		double upper = maximumLocation + width;
		RandomEngine commonRandom = random.clone();
		double lowerProbability = probability(lower, location, scale,
				degreesOfFreedom, options, commonRandom);
		double upperProbability = probability(upper, location, scale,
				degreesOfFreedom, options, commonRandom);
		for (int expansion = 0; expansion < 24 &&
				(lowerProbability > p || upperProbability < p); expansion++) {
			width *= 2.0;
			if (lowerProbability > p) {
				lower = minimumLocation - width;
				lowerProbability = probability(lower, location, scale,
						degreesOfFreedom, options, commonRandom);
			}
			if (upperProbability < p) {
				upper = maximumLocation + width;
				upperProbability = probability(upper, location, scale,
						degreesOfFreedom, options, commonRandom);
			}
		}
		if (!(lowerProbability <= p) || !(upperProbability >= p))
			return Double.NaN;
		for (int iteration = 0; iteration < 56; iteration++) {
			double middle = lower + 0.5 * (upper - lower);
			double middleProbability = probability(middle, location, scale,
					degreesOfFreedom, options, commonRandom);
			if (Double.isNaN(middleProbability)) return Double.NaN;
			if (middleProbability < p) lower = middle;
			else upper = middle;
			if (upper - lower <= 8.0 * Math.ulp(Math.max(1.0, Math.abs(middle))))
				break;
		}
		return lower + 0.5 * (upper - lower);
	}

	private static double probability(double upper, double[] location,
			double[][] scale, double degreesOfFreedom,
			MultivariateProbabilityOptions options, RandomEngine commonRandom) {
		double[] lower = new double[location.length];
		double[] upperVector = new double[location.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		Arrays.fill(upperVector, upper);
		MultivariateProbabilityResult result = degreesOfFreedom > 0.0
				? MultivariateProbability.studentT(lower, upperVector, location, scale,
						degreesOfFreedom, options, commonRandom.clone())
				: MultivariateProbability.normal(lower, upperVector, location, scale,
						options, commonRandom.clone());
		return result.probability;
	}
}
