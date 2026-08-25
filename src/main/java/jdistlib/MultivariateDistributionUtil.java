/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Internal validation and Cholesky operations for vector distributions. */
final class MultivariateDistributionUtil {
	private MultivariateDistributionUtil() {}

	static final class Factor {
		final double[][] lower;
		final double logDeterminant;

		Factor(double[][] lower, double logDeterminant) {
			this.lower = lower;
			this.logDeterminant = logDeterminant;
		}
	}

	static Factor factor(double[] location, double[][] scatter) {
		if (location == null || scatter == null || location.length == 0 ||
				scatter.length != location.length) return null;
		for (double value : location)
			if (!Double.isFinite(value)) return null;
		return factor(scatter, location.length);
	}

	static Factor factor(double[][] scatter) {
		return factor(scatter, scatter == null ? 0 : scatter.length);
	}

	private static Factor factor(double[][] scatter, int dimension) {
		if (scatter == null || dimension == 0 || scatter.length != dimension)
			return null;
		for (int i = 0; i < dimension; i++)
			if (scatter[i] == null || scatter[i].length != dimension) return null;
		for (int i = 0; i < dimension; i++) {
			for (int j = 0; j < dimension; j++) {
				double value = scatter[i][j];
				if (!Double.isFinite(value)) return null;
				double tolerance = 1e-12 * Math.max(1.0,
						Math.max(Math.abs(value), Math.abs(scatter[j][i])));
				if (Math.abs(value - scatter[j][i]) > tolerance) return null;
			}
		}

		double[][] lower = new double[dimension][dimension];
		double logDeterminant = 0.0;
		for (int i = 0; i < dimension; i++) {
			for (int j = 0; j <= i; j++) {
				double sum = scatter[i][j];
				for (int k = 0; k < j; k++) sum -= lower[i][k] * lower[j][k];
				if (i == j) {
					if (!(sum > 0.0) || !Double.isFinite(sum)) return null;
					lower[i][j] = Math.sqrt(sum);
					logDeterminant += Math.log(sum);
				} else {
					lower[i][j] = sum / lower[j][j];
				}
			}
		}
		return new Factor(lower, logDeterminant);
	}

	static double quadratic(double[] x, double[] location, Factor factor) {
		if (x == null || x.length != location.length) return Double.NaN;
		double sumSquares = 0.0;
		double[] solved = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			if (Double.isNaN(x[i])) return Double.NaN;
			if (Double.isInfinite(x[i])) return Double.POSITIVE_INFINITY;
			double value = x[i] - location[i];
			for (int j = 0; j < i; j++) value -= factor.lower[i][j] * solved[j];
			solved[i] = value / factor.lower[i][i];
			sumSquares += solved[i] * solved[i];
		}
		return sumSquares;
	}

	static double[] transform(double[] location, Factor factor, double[] z,
			double multiplier) {
		double[] result = new double[location.length];
		for (int i = 0; i < result.length; i++) {
			double value = 0.0;
			for (int j = 0; j <= i; j++) value += factor.lower[i][j] * z[j];
			result[i] = location[i] + multiplier * value;
		}
		return result;
	}

	static double[] standardNormal(int dimension, RandomEngine random) {
		double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) result[i] = random.nextGaussian();
		return result;
	}

	static double logAdd(double left, double right) {
		if (left == Double.NEGATIVE_INFINITY) return right;
		if (right == Double.NEGATIVE_INFINITY) return left;
		double maximum = Math.max(left, right);
		return maximum + Math.log(Math.exp(left - maximum) + Math.exp(right - maximum));
	}
}
