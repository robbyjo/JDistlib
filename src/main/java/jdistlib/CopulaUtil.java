/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Package-private copula validation and numerical helpers. */
final class CopulaUtil {
	private CopulaUtil() {}

	static void requireDimension(int dimension) {
		if (dimension < 1) throw new IllegalArgumentException("dimension must be positive");
	}

	static void requireMultivariateDimension(int dimension) {
		if (dimension < 2) throw new IllegalArgumentException("copula dimension must be at least two");
	}

	static boolean validPoint(double[] u, int dimension) {
		return CopulaDiagnostics.inspect(u, dimension).isValid();
	}

	static boolean interiorPoint(double[] u, int dimension) {
		return CopulaDiagnostics.inspect(u, dimension).isInterior();
	}

	static boolean hasZero(double[] u) {
		for (double value : u) if (value == 0.0) return true;
		return false;
	}

	static void requirePair(int first, int second, int dimension) {
		if (first < 0 || second < 0 || first >= dimension || second >= dimension)
			throw new IndexOutOfBoundsException("copula coordinate index out of range");
	}

	static double uniformOpen(RandomEngine random) {
		return clampOpen(random.nextDouble());
	}

	static double clampOpen(double value) {
		if (value <= 0.0) return Math.nextUp(0.0);
		if (value >= 1.0) return Math.nextDown(1.0);
		return value;
	}

	static double[][] copyMatrix(double[][] matrix) {
		double[][] result = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) result[i] = matrix[i].clone();
		return result;
	}

	static double[] zeros(int dimension) { return new double[dimension]; }

	static double logSumExp(double[] values) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (double value : values) maximum = Math.max(maximum, value);
		if (Double.isInfinite(maximum)) return maximum;
		double sum = 0.0;
		for (double value : values) sum += Math.exp(value - maximum);
		return maximum + Math.log(sum);
	}

	static void validateCorrelation(double[][] correlation) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(correlation);
		if (factor == null) {
			throw new IllegalArgumentException(
					"correlation must be a finite symmetric positive-definite matrix");
		}
		for (int i = 0; i < correlation.length; i++) {
			double tolerance = 1e-12 * Math.max(1.0, Math.abs(correlation[i][i]));
			if (Math.abs(correlation[i][i] - 1.0) > tolerance) {
				throw new IllegalArgumentException("correlation diagonal must equal one");
			}
		}
	}
}
