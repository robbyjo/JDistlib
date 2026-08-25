/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Elliptical multivariate power-exponential (generalized Gaussian) law. */
public final class MultivariatePowerExponential {
	private MultivariatePowerExponential() {}

	public static double density(double[] x, double[] location, double[][] scatter,
			double shape, boolean giveLog) {
		if (!(shape > 0.0) || !Double.isFinite(shape)) return Double.NaN;
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, scatter);
		if (factor == null) return Double.NaN;
		double quadratic = MultivariateDistributionUtil.quadratic(x, location, factor);
		if (Double.isNaN(quadratic)) return Double.NaN;
		int dimension = location.length;
		double radialShape = dimension / (2.0 * shape);
		double logDensity = Math.log(shape) + lgammafn(dimension / 2.0) -
				0.5 * dimension * Math.log(Math.PI) - radialShape * Math.log(2.0) -
				lgammafn(radialShape) - 0.5 * factor.logDeterminant -
				0.5 * Math.pow(quadratic, shape);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] location, double[][] scatter,
			double shape, RandomEngine random) {
		if (!(shape > 0.0) || !Double.isFinite(shape)) return null;
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, scatter);
		if (factor == null || random == null) return null;
		double[] direction;
		double norm;
		do {
			direction = MultivariateDistributionUtil.standardNormal(location.length, random);
			norm = 0.0;
			for (double value : direction) norm += value * value;
			norm = Math.sqrt(norm);
		} while (norm == 0.0);
		for (int i = 0; i < direction.length; i++) direction[i] /= norm;
		double gamma = Gamma.random(location.length / (2.0 * shape), 1.0, random);
		double radius = Math.pow(2.0 * gamma, 1.0 / (2.0 * shape));
		return MultivariateDistributionUtil.transform(location, factor, direction, radius);
	}

	public static double[][] random(int n, double[] location, double[][] scatter,
			double shape, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++)
			result[i] = random(location, scatter, shape, random);
		return result;
	}

	/** Quantile of the scatter-standardized radial distance. */
	public static double radialQuantile(double p, int dimension, double shape,
			boolean lowerTail, boolean logProbability) {
		if (dimension < 1 || !(shape > 0.0) || !Double.isFinite(shape))
			return Double.NaN;
		double gamma = Gamma.quantile(p, dimension / (2.0 * shape), 1.0,
				lowerTail, logProbability);
		return Math.pow(2.0 * gamma, 1.0 / (2.0 * shape));
	}
}
