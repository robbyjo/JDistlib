/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lmvgammafn;

import jdistlib.rng.RandomEngine;

/** Wishart distribution on symmetric positive-definite matrices. */
public final class Wishart {
	private Wishart() {}

	/**
	 * Density of {@code W_dimension(scale, degreesOfFreedom)}.
	 * The mean under this parameterization is {@code degreesOfFreedom * scale}.
	 */
	public static double density(double[][] x, double degreesOfFreedom,
			double[][] scale, boolean giveLog) {
		MultivariateDistributionUtil.Factor scaleFactor =
				MultivariateDistributionUtil.factor(scale);
		if (scaleFactor == null) return Double.NaN;
		return densityWithFactor(x, degreesOfFreedom, scaleFactor, giveLog);
	}

	/** Density overload accepting the lower Cholesky factor {@code L} of scale. */
	public static double densityFromCholesky(double[][] x,
			double degreesOfFreedom, double[][] lowerCholesky, boolean giveLog) {
		MultivariateDistributionUtil.Factor scaleFactor =
				factorFromCholesky(lowerCholesky);
		if (scaleFactor == null) return Double.NaN;
		return densityWithFactor(x, degreesOfFreedom, scaleFactor, giveLog);
	}

	private static double densityWithFactor(double[][] x,
			double degreesOfFreedom, MultivariateDistributionUtil.Factor scaleFactor,
			boolean giveLog) {
		int dimension = scaleFactor.lower.length;
		if (!(degreesOfFreedom > dimension - 1.0) ||
				!Double.isFinite(degreesOfFreedom)) return Double.NaN;
		if (!squareFiniteSymmetric(x, dimension)) return Double.NaN;
		MultivariateDistributionUtil.Factor xFactor =
				MultivariateDistributionUtil.factor(x);
		if (xFactor == null)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;

		double trace = 0.0;
		for (int column = 0; column < dimension; column++) {
			double[] solved = new double[dimension];
			for (int row = 0; row < dimension; row++) {
				double value = row >= column ? xFactor.lower[row][column] : 0.0;
				for (int k = 0; k < row; k++)
					value -= scaleFactor.lower[row][k] * solved[k];
				solved[row] = value / scaleFactor.lower[row][row];
				trace += solved[row] * solved[row];
			}
		}
		double logDensity = 0.5 * (degreesOfFreedom - dimension - 1.0) *
				xFactor.logDeterminant - 0.5 * trace -
				0.5 * degreesOfFreedom * dimension * Math.log(2.0) -
				0.5 * degreesOfFreedom * scaleFactor.logDeterminant -
				lmvgammafn(0.5 * degreesOfFreedom, dimension);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	/**
	 * Generates one Wishart matrix using Bartlett's decomposition.
	 * {@code lowerCholesky} is a lower-triangular L with {@code scale = L L'}.
	 */
	public static double[][] random(double degreesOfFreedom,
			double[][] lowerCholesky, RandomEngine random) {
		MultivariateDistributionUtil.Factor scaleFactor =
				factorFromCholesky(lowerCholesky);
		if (scaleFactor == null || random == null) return null;
		int dimension = lowerCholesky.length;
		if (!(degreesOfFreedom > dimension - 1.0) ||
				!Double.isFinite(degreesOfFreedom)) return null;
		double[][] bartlett = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) {
			bartlett[i][i] = Math.sqrt(ChiSquare.random(degreesOfFreedom - i,
					random));
			for (int j = 0; j < i; j++) bartlett[i][j] = random.nextGaussian();
		}
		double[][] product = multiplyLower(lowerCholesky, bartlett);
		return crossProduct(product);
	}

	/** Generates {@code count} matrices while retaining one caller-owned RNG. */
	public static double[][][] random(int count, double degreesOfFreedom,
			double[][] lowerCholesky, RandomEngine random) {
		if (count < 0) return null;
		double[][][] result = new double[count][][];
		for (int i = 0; i < count; i++) {
			result[i] = random(degreesOfFreedom, lowerCholesky, random);
			if (result[i] == null) return null;
		}
		return result;
	}

	/** Convenience generator accepting the scale matrix rather than its factor. */
	public static double[][] randomFromScale(double degreesOfFreedom,
			double[][] scale, RandomEngine random) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(scale);
		return factor == null ? null : random(degreesOfFreedom, factor.lower, random);
	}

	public static double[][][] randomFromScale(int count, double degreesOfFreedom,
			double[][] scale, RandomEngine random) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(scale);
		return factor == null ? null : random(count, degreesOfFreedom, factor.lower,
				random);
	}

	private static MultivariateDistributionUtil.Factor factorFromCholesky(
			double[][] lower) {
		if (lower == null || lower.length == 0) return null;
		int dimension = lower.length;
		double logDeterminant = 0.0;
		double[][] copy = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) {
			if (lower[i] == null || lower[i].length != dimension) return null;
			for (int j = 0; j < dimension; j++) {
				if (!Double.isFinite(lower[i][j]) || (j > i && lower[i][j] != 0.0))
					return null;
				copy[i][j] = lower[i][j];
			}
			if (!(copy[i][i] > 0.0)) return null;
			logDeterminant += 2.0 * Math.log(copy[i][i]);
		}
		return new MultivariateDistributionUtil.Factor(copy, logDeterminant);
	}

	private static boolean squareFiniteSymmetric(double[][] matrix, int dimension) {
		if (matrix == null || matrix.length != dimension) return false;
		for (int i = 0; i < dimension; i++)
			if (matrix[i] == null || matrix[i].length != dimension) return false;
		for (int i = 0; i < dimension; i++) {
			for (int j = 0; j < dimension; j++) {
				if (!Double.isFinite(matrix[i][j])) return false;
				double tolerance = 1e-12 * Math.max(1.0,
						Math.max(Math.abs(matrix[i][j]), Math.abs(matrix[j][i])));
				if (Math.abs(matrix[i][j] - matrix[j][i]) > tolerance) return false;
			}
		}
		return true;
	}

	private static double[][] multiplyLower(double[][] left, double[][] right) {
		int dimension = left.length;
		double[][] result = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++)
			for (int j = 0; j <= i; j++)
				for (int k = j; k <= i; k++)
					result[i][j] += left[i][k] * right[k][j];
		return result;
	}

	private static double[][] crossProduct(double[][] lower) {
		int dimension = lower.length;
		double[][] result = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) {
			for (int j = 0; j <= i; j++) {
				double sum = 0.0;
				for (int k = 0; k <= Math.min(i, j); k++)
					sum += lower[i][k] * lower[j][k];
				result[i][j] = result[j][i] = sum;
			}
		}
		return result;
	}
}
