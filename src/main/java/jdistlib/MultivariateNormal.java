/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Multivariate normal density and random generation using a covariance matrix. */
public final class MultivariateNormal {
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
}
