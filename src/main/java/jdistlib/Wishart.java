/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lmvgammafn;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Wishart distribution on symmetric positive-definite matrices. */
public final class Wishart {
	private static final long PROBABILITY_SEED = 0x5769736861727450L;
	private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
	private static final double LOG_MIN_VALUE = Math.log(Double.MIN_VALUE);

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

	/**
	 * CDF of the directional variance event {@code direction' W direction <= upper}.
	 * For {@code W ~ Wishart(df, scale)}, the quadratic form divided by
	 * {@code direction' scale direction} is exactly chi-square with {@code df}
	 * degrees of freedom.
	 */
	public static double quadraticFormCumulative(double upper, double[] direction,
			double degreesOfFreedom, double[][] scale, boolean lowerTail,
			boolean logProbability) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(scale);
		if (factor == null || direction == null ||
				direction.length != factor.lower.length ||
				!(degreesOfFreedom > direction.length - 1.0) ||
				!Double.isFinite(degreesOfFreedom)) return Double.NaN;
		double scaleValue = 0.0;
		boolean nonzero = false;
		for (int i = 0; i < direction.length; i++) {
			if (!Double.isFinite(direction[i])) return Double.NaN;
			nonzero |= direction[i] != 0.0;
			double transformed = 0.0;
			for (int j = i; j < direction.length; j++)
				transformed += factor.lower[j][i] * direction[j];
			scaleValue += transformed * transformed;
		}
		if (!nonzero || !(scaleValue > 0.0)) return Double.NaN;
		return ChiSquare.cumulative(upper / scaleValue, degreesOfFreedom,
				lowerTail, logProbability);
	}

	/**
	 * CDF of {@code trace(scale^-1 W)}. This standardized trace is exactly
	 * chi-square with {@code dimension * degreesOfFreedom} degrees of freedom.
	 */
	public static double standardizedTraceCumulative(double upper,
			double degreesOfFreedom, double[][] scale, boolean lowerTail,
			boolean logProbability) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(scale);
		if (factor == null ||
				!(degreesOfFreedom > factor.lower.length - 1.0) ||
				!Double.isFinite(degreesOfFreedom)) return Double.NaN;
		return ChiSquare.cumulative(upper,
				degreesOfFreedom * factor.lower.length, lowerTail, logProbability);
	}

	/**
	 * Computes {@code P(lower <= determinant(W) <= upper)}. The numerical path
	 * conditions on Bartlett's independent chi-square determinant factors and
	 * reports randomized-integration error metadata.
	 */
	public static MultivariateProbabilityResult determinantProbability(
			double lower, double upper, double degreesOfFreedom, double[][] scale,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (Double.isNaN(lower) || Double.isNaN(upper)) return invalidProbability();
		if (!(lower < upper) || upper <= 0.0) return exactProbability(0.0);
		double logLower = lower <= 0.0 ? Double.NEGATIVE_INFINITY : Math.log(lower);
		double logUpper = upper == Double.POSITIVE_INFINITY
				? Double.POSITIVE_INFINITY : Math.log(upper);
		return logDeterminantProbability(logLower, logUpper, degreesOfFreedom,
				scale, options, random);
	}

	public static MultivariateProbabilityResult determinantProbability(
			double lower, double upper, double degreesOfFreedom, double[][] scale) {
		return determinantProbability(lower, upper, degreesOfFreedom, scale,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P(determinant(W) <= upper)}. */
	public static MultivariateProbabilityResult determinantCumulative(double upper,
			double degreesOfFreedom, double[][] scale,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return determinantProbability(0.0, upper, degreesOfFreedom, scale, options,
				random);
	}

	public static MultivariateProbabilityResult determinantCumulative(double upper,
			double degreesOfFreedom, double[][] scale) {
		return determinantCumulative(upper, degreesOfFreedom, scale,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/**
	 * Computes {@code P(lower <= log(det(W)) <= upper)} without exponentiating
	 * the caller's thresholds, so extreme determinant events remain representable.
	 */
	public static MultivariateProbabilityResult logDeterminantProbability(
			final double lower, final double upper, final double degreesOfFreedom,
			double[][] scale, MultivariateProbabilityOptions options,
			RandomEngine random) {
		final MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(scale);
		if (factor == null || Double.isNaN(lower) || Double.isNaN(upper) ||
				!(degreesOfFreedom > factor.lower.length - 1.0) ||
				!Double.isFinite(degreesOfFreedom)) return invalidProbability();
		if (!(lower < upper)) return exactProbability(0.0);
		if (lower == Double.NEGATIVE_INFINITY &&
				upper == Double.POSITIVE_INFINITY) return exactProbability(1.0);
		final int dimension = factor.lower.length;
		final double standardizedLower = lower - factor.logDeterminant;
		final double standardizedUpper = upper - factor.logDeterminant;
		if (dimension == 1)
			return exactProbability(chiSquareLogInterval(standardizedLower,
					standardizedUpper, degreesOfFreedom));
		MultivariateProbabilityResult estimate = MultivariateProbability.integrate(
				new MultivariateProbability.Integrand() {
			@Override public int dimension() { return dimension - 1; }
			@Override public double value(double[] uniforms) {
				double partialLogDeterminant = 0.0;
				for (int i = 0; i + 1 < dimension; i++) {
					double factorValue = ChiSquare.quantile(uniforms[i],
							degreesOfFreedom - i, true, false);
					if (!(factorValue > 0.0) || !Double.isFinite(factorValue)) return 0.0;
					partialLogDeterminant += Math.log(factorValue);
				}
				return chiSquareLogInterval(standardizedLower - partialLogDeterminant,
						standardizedUpper - partialLogDeterminant,
						degreesOfFreedom - dimension + 1.0);
			}
		}, options, random);
		// Constant randomized samples cannot certify a finite-threshold tail event.
		if ((estimate.probability == 0.0 || estimate.probability == 1.0) &&
				estimate.evaluations > 0)
			return new MultivariateProbabilityResult(estimate.probability,
					Math.max(estimate.absoluteError, 1.0 / estimate.evaluations),
					estimate.evaluations, 1);
		return estimate;
	}

	public static MultivariateProbabilityResult logDeterminantProbability(
			double lower, double upper, double degreesOfFreedom, double[][] scale) {
		return logDeterminantProbability(lower, upper, degreesOfFreedom, scale,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P(log(det(W)) <= upper)}. */
	public static MultivariateProbabilityResult logDeterminantCumulative(
			double upper, double degreesOfFreedom, double[][] scale,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return logDeterminantProbability(Double.NEGATIVE_INFINITY, upper,
				degreesOfFreedom, scale, options, random);
	}

	public static MultivariateProbabilityResult logDeterminantCumulative(
			double upper, double degreesOfFreedom, double[][] scale) {
		return logDeterminantCumulative(upper, degreesOfFreedom, scale,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	private static double chiSquareLogInterval(double lower, double upper,
			double degreesOfFreedom) {
		if (!(lower < upper)) return 0.0;
		double lowerValue = lower <= LOG_MIN_VALUE ? 0.0 :
				(lower >= LOG_MAX_VALUE ? Double.POSITIVE_INFINITY : Math.exp(lower));
		double upperValue = upper <= LOG_MIN_VALUE ? 0.0 :
				(upper >= LOG_MAX_VALUE ? Double.POSITIVE_INFINITY : Math.exp(upper));
		if (!(lowerValue < upperValue)) return 0.0;
		if (lowerValue >= degreesOfFreedom) {
			return Math.max(0.0, ChiSquare.cumulative(lowerValue, degreesOfFreedom,
					false, false) - ChiSquare.cumulative(upperValue, degreesOfFreedom,
					false, false));
		}
		return Math.max(0.0, ChiSquare.cumulative(upperValue, degreesOfFreedom,
				true, false) - ChiSquare.cumulative(lowerValue, degreesOfFreedom,
				true, false));
	}

	private static MultivariateProbabilityResult exactProbability(double value) {
		return new MultivariateProbabilityResult(value, 0.0, 0, 0);
	}

	private static MultivariateProbabilityResult invalidProbability() {
		return new MultivariateProbabilityResult(Double.NaN, Double.NaN, 0, 2);
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
