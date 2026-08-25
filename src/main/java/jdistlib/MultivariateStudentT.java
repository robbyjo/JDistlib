/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Elliptical multivariate Student t distribution. */
public final class MultivariateStudentT {
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
}
