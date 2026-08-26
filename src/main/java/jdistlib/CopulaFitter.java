/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Rank transformation and dependence fitting for the built-in copula families. */
public final class CopulaFitter {
	private CopulaFitter() {}

	/** Converts rectangular raw data to average-rank pseudo-observations. */
	public static double[][] pseudoObservations(double[][] data) {
		int dimension = validateRectangular(data, false);
		int count = data.length;
		double[][] result = new double[count][dimension];
		for (int coordinate = 0; coordinate < dimension; coordinate++) {
			Integer[] order = new Integer[count];
			for (int i = 0; i < count; i++) order[i] = i;
			final int column = coordinate;
			Arrays.sort(order, (left, right) ->
					Double.compare(data[left][column], data[right][column]));
			int start = 0;
			while (start < count) {
				int end = start + 1;
				while (end < count && data[order[start]][coordinate]
						== data[order[end]][coordinate]) end++;
				double averageRank = 0.5 * ((start + 1.0) + end);
				double probability = averageRank / (count + 1.0);
				for (int i = start; i < end; i++) result[order[i]][coordinate] = probability;
				start = end;
			}
		}
		return result;
	}

	/**
	 * Applies marginal probability transforms. Discrete coordinates use the
	 * midpoint of their CDF jump, producing a deterministic mixed-data transform.
	 */
	public static double[][] marginalTransforms(double[][] data,
			CopulaMarginal... marginals) {
		return marginalTransforms(data, null, marginals);
	}

	/**
	 * Applies marginal probability transforms. With a random engine, each
	 * discrete CDF jump is randomized uniformly; with {@code null}, its midpoint
	 * is used.
	 */
	public static double[][] marginalTransforms(double[][] data, RandomEngine random,
			CopulaMarginal... marginals) {
		int dimension = validateRectangular(data, false);
		if (marginals == null || marginals.length != dimension)
			throw new IllegalArgumentException("one declared marginal is required per column");
		for (CopulaMarginal marginal : marginals)
			if (marginal == null) throw new IllegalArgumentException("marginals must not contain null");
		double[][] result = new double[data.length][dimension];
		for (int row = 0; row < data.length; row++) {
			for (int column = 0; column < dimension; column++) {
				double upper = marginals[column].cumulative(data[row][column]);
				double lower = marginals[column].leftCumulative(data[row][column]);
				double weight = marginals[column].isDiscrete()
						? (random == null ? 0.5 : CopulaUtil.uniformOpen(random)) : 1.0;
				double transformed = lower + weight * (upper - lower);
				if (!Double.isFinite(transformed) || transformed < 0.0 || transformed > 1.0)
					throw new IllegalArgumentException("marginal transform produced an invalid probability");
				result[row][column] = CopulaUtil.clampOpen(transformed);
			}
		}
		return result;
	}

	/** Applies a reproducible randomized distributional transform. */
	public static double[][] marginalTransforms(double[][] data, long seed,
			CopulaMarginal... marginals) {
		return marginalTransforms(data, new MersenneTwister(seed), marginals);
	}

	public static CopulaFitResult fit(double[][] data, CopulaFamily family) {
		return fit(data, family, new CopulaFitOptions());
	}

	/** Fits raw observations after applying marginal ranks. */
	public static CopulaFitResult fit(double[][] data, CopulaFamily family,
			CopulaFitOptions options) {
		if (family == null || options == null)
			return failure(family, CopulaFitResult.Status.INVALID_DATA,
					"family and options must not be null", 0);
		try {
			return fitUniforms(pseudoObservations(data), family, options);
		} catch (IllegalArgumentException exception) {
			return failure(family, CopulaFitResult.Status.INVALID_DATA,
					exception.getMessage(), data == null ? 0 : data.length);
		}
	}

	public static CopulaFitResult fitUniforms(double[][] uniforms,
			CopulaFamily family) {
		return fitUniforms(uniforms, family, new CopulaFitOptions());
	}

	/** Fits declared continuous/discrete marginals using midpoint transforms. */
	public static CopulaFitResult fitMixed(double[][] data,
			CopulaMarginal[] marginals, CopulaFamily family) {
		return fitMixed(data, marginals, null, family, new CopulaFitOptions());
	}

	/** Fits declared marginals using a reproducible randomized transform. */
	public static CopulaFitResult fitMixed(double[][] data,
			CopulaMarginal[] marginals, long seed, CopulaFamily family) {
		return fitMixed(data, marginals, new MersenneTwister(seed), family,
				new CopulaFitOptions());
	}

	/** Fits declared marginals using a reproducible randomized transform. */
	public static CopulaFitResult fitMixed(double[][] data,
			CopulaMarginal[] marginals, long seed, CopulaFamily family,
			CopulaFitOptions options) {
		return fitMixed(data, marginals, new MersenneTwister(seed), family,
				options);
	}

	/** Fits declared marginals using midpoint or randomized distributional transforms. */
	public static CopulaFitResult fitMixed(double[][] data,
			CopulaMarginal[] marginals, RandomEngine random, CopulaFamily family,
			CopulaFitOptions options) {
		try {
			return fitUniforms(marginalTransforms(data, random, marginals), family, options);
		} catch (IllegalArgumentException exception) {
			return failure(family, CopulaFitResult.Status.INVALID_DATA,
					exception.getMessage(), data == null ? 0 : data.length);
		}
	}

	/** Fits observations already transformed to the open unit hypercube. */
	public static CopulaFitResult fitUniforms(double[][] uniforms,
			CopulaFamily family, CopulaFitOptions options) {
		if (family == null || options == null)
			return failure(family, CopulaFitResult.Status.INVALID_DATA,
					"family and options must not be null", 0);
		final int dimension;
		try {
			dimension = validateRectangular(uniforms, true);
		} catch (IllegalArgumentException exception) {
			return failure(family, CopulaFitResult.Status.INVALID_DATA,
					exception.getMessage(), uniforms == null ? 0 : uniforms.length);
		}
		double[][] tau = kendallsTau(uniforms);
		double averageTau = averageOffDiagonal(tau);
		try {
			switch (family) {
			case INDEPENDENCE:
				return success(family, new IndependenceCopula(dimension), uniforms, 0);
			case GAUSSIAN:
				double[][] gaussianCorrelation = options.getMethod()
						== CopulaFitOptions.Method.MAXIMUM_LIKELIHOOD
						? normalScoreCorrelation(uniforms)
						: GaussianCopula.correlationFromKendallsTau(tau);
				return success(family, new GaussianCopula(
						shrinkToPositiveDefinite(gaussianCorrelation)), uniforms,
						dimension * (dimension - 1) / 2);
			case STUDENT_T:
				double[][] studentCorrelation = shrinkToPositiveDefinite(
						GaussianCopula.correlationFromKendallsTau(tau));
				double df = optimizeDegreesOfFreedom(uniforms, studentCorrelation, options);
				return success(family, new StudentTCopula(studentCorrelation, df),
						uniforms, dimension * (dimension - 1) / 2 + 1);
			case CLAYTON:
				if (averageTau < 0.0) return incompatible(family, uniforms.length,
						"Clayton cannot represent negative average dependence");
				double clayton = ClaytonCopula.parameterFromKendallsTau(
						Math.min(0.999999, averageTau));
				if (options.getMethod() == CopulaFitOptions.Method.MAXIMUM_LIKELIHOOD)
					clayton = optimizeScalar(uniforms, 0.0,
							Math.max(30.0, clayton * 4.0 + 2.0), options,
							value -> new ClaytonCopula(dimension, value));
				return success(family, new ClaytonCopula(dimension, clayton), uniforms, 1);
			case GUMBEL:
				if (averageTau < 0.0) return incompatible(family, uniforms.length,
						"Gumbel cannot represent negative average dependence");
				double gumbel = GumbelCopula.parameterFromKendallsTau(
						Math.min(0.999999, averageTau));
				if (options.getMethod() == CopulaFitOptions.Method.MAXIMUM_LIKELIHOOD)
					gumbel = optimizeScalar(uniforms, 1.0,
							Math.max(30.0, gumbel * 4.0 + 2.0), options,
							value -> new GumbelCopula(dimension, value));
				return success(family, new GumbelCopula(dimension, gumbel), uniforms, 1);
			case FRANK:
				if (dimension > 2 && averageTau < 0.0)
					return incompatible(family, uniforms.length,
							"negative Frank dependence is only bivariate");
				double frank = FrankCopula.parameterFromKendallsTau(
						Math.max(-0.999999, Math.min(0.999999, averageTau)));
				if (options.getMethod() == CopulaFitOptions.Method.MAXIMUM_LIKELIHOOD) {
					double lower = dimension == 2 ? -50.0 : 0.0;
					frank = optimizeScalar(uniforms, lower, 50.0, options,
							value -> new FrankCopula(dimension, value));
				}
				return success(family, new FrankCopula(dimension, frank), uniforms, 1);
			default:
				throw new AssertionError(family);
			}
		} catch (RuntimeException exception) {
			return failure(family, CopulaFitResult.Status.NUMERICAL_FAILURE,
					exception.getMessage(), uniforms.length);
		}
	}

	/** Pairwise empirical Kendall tau computed over untied pairs. */
	public static double[][] kendallsTau(double[][] uniforms) {
		int dimension = validateRectangular(uniforms, false);
		double[][] result = new double[dimension][dimension];
		for (int first = 0; first < dimension; first++) {
			result[first][first] = 1.0;
			for (int second = first + 1; second < dimension; second++) {
				long concordant = 0;
				long discordant = 0;
				for (int i = 0; i < uniforms.length; i++) {
					for (int j = i + 1; j < uniforms.length; j++) {
						double product = (uniforms[i][first] - uniforms[j][first])
								* (uniforms[i][second] - uniforms[j][second]);
						if (product > 0.0) concordant++;
						else if (product < 0.0) discordant++;
					}
				}
				double tau = concordant + discordant == 0 ? 0.0
						: (double) (concordant - discordant) / (concordant + discordant);
				result[first][second] = tau;
				result[second][first] = tau;
			}
		}
		return result;
	}

	private static double optimizeDegreesOfFreedom(double[][] uniforms,
			double[][] correlation, CopulaFitOptions options) {
		double low = Math.log(options.getMinimumDegreesOfFreedom());
		double high = Math.log(options.getMaximumDegreesOfFreedom());
		double optimum = goldenMaximum(low, high, options.getOptimizationIterations(),
				value -> logLikelihood(new StudentTCopula(correlation, Math.exp(value)), uniforms));
		return Math.exp(optimum);
	}

	private static double optimizeScalar(double[][] uniforms, double low,
			double high, CopulaFitOptions options, CopulaFactory factory) {
		return goldenMaximum(low, high, options.getOptimizationIterations(),
				value -> logLikelihood(factory.create(value), uniforms));
	}

	private static double goldenMaximum(double low, double high, int iterations,
			Objective objective) {
		double ratio = (Math.sqrt(5.0) - 1.0) / 2.0;
		double left = high - ratio * (high - low);
		double right = low + ratio * (high - low);
		double leftValue = objective.value(left);
		double rightValue = objective.value(right);
		for (int i = 0; i < iterations; i++) {
			if (leftValue > rightValue) {
				high = right;
				right = left;
				rightValue = leftValue;
				left = high - ratio * (high - low);
				leftValue = objective.value(left);
			} else {
				low = left;
				left = right;
				leftValue = rightValue;
				right = low + ratio * (high - low);
				rightValue = objective.value(right);
			}
		}
		return leftValue > rightValue ? left : right;
	}

	private static double[][] normalScoreCorrelation(double[][] uniforms) {
		int count = uniforms.length;
		int dimension = uniforms[0].length;
		double[][] scores = new double[count][dimension];
		double[] mean = new double[dimension];
		for (int i = 0; i < count; i++) {
			for (int j = 0; j < dimension; j++) {
				scores[i][j] = Normal.quantile(uniforms[i][j], 0.0, 1.0, true, false);
				mean[j] += scores[i][j] / count;
			}
		}
		double[][] covariance = new double[dimension][dimension];
		for (double[] score : scores) {
			for (int i = 0; i < dimension; i++) {
				for (int j = 0; j <= i; j++)
					covariance[i][j] += (score[i] - mean[i]) * (score[j] - mean[j]);
			}
		}
		double[][] correlation = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) {
			correlation[i][i] = 1.0;
			for (int j = 0; j < i; j++) {
				double value = covariance[i][j]
						/ Math.sqrt(covariance[i][i] * covariance[j][j]);
				value = Math.max(-0.999999, Math.min(0.999999, value));
				correlation[i][j] = value;
				correlation[j][i] = value;
			}
		}
		return correlation;
	}

	private static double[][] shrinkToPositiveDefinite(double[][] correlation) {
		double shrink = 1.0;
		for (int attempt = 0; attempt < 100; attempt++) {
			double[][] candidate = new double[correlation.length][correlation.length];
			for (int i = 0; i < correlation.length; i++) {
				candidate[i][i] = 1.0;
				for (int j = 0; j < i; j++) {
					candidate[i][j] = shrink * correlation[i][j];
					candidate[j][i] = candidate[i][j];
				}
			}
			if (MultivariateDistributionUtil.factor(candidate) != null) return candidate;
			shrink *= 0.95;
		}
		throw new IllegalArgumentException("could not regularize the fitted correlation matrix");
	}

	private static CopulaFitResult success(CopulaFamily family, Copula copula,
			double[][] uniforms, int parameters) {
		CopulaLikelihoodDiagnostics diagnostics =
				CopulaLikelihoodDiagnostics.assess(copula, uniforms);
		double likelihood = diagnostics.getLogLikelihood();
		if (!diagnostics.isSuccess()) {
			return new CopulaFitResult(family, null, Double.NaN, uniforms.length,
					0, CopulaFitResult.Status.NUMERICAL_FAILURE,
					diagnostics.message(), diagnostics);
		}
		return new CopulaFitResult(family, copula, likelihood, uniforms.length,
				parameters, CopulaFitResult.Status.SUCCESS, "fit completed",
				diagnostics);
	}

	private static CopulaFitResult incompatible(CopulaFamily family, int count,
			String message) {
		return failure(family, CopulaFitResult.Status.INCOMPATIBLE_DEPENDENCE,
				message, count);
	}

	private static CopulaFitResult failure(CopulaFamily family,
			CopulaFitResult.Status status, String message, int count) {
		return new CopulaFitResult(family, null, Double.NaN, count, 0, status,
				message == null ? status.name() : message,
				CopulaLikelihoodDiagnostics.invalid(count,
						message == null ? status.name() : message));
	}

	private static double logLikelihood(Copula copula, double[][] uniforms) {
		double sum = 0.0;
		for (double[] uniform : uniforms) {
			double value = copula.logDensity(uniform);
			if (!Double.isFinite(value)) return Double.NEGATIVE_INFINITY;
			sum += value;
		}
		return sum;
	}

	private static double averageOffDiagonal(double[][] matrix) {
		double sum = 0.0;
		int count = 0;
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < i; j++) { sum += matrix[i][j]; count++; }
		}
		return sum / count;
	}

	private static int validateRectangular(double[][] data, boolean requireUnit) {
		if (data == null || data.length < 2 || data[0] == null || data[0].length < 2)
			throw new IllegalArgumentException("data need at least two rows and two columns");
		int dimension = data[0].length;
		for (double[] row : data) {
			if (row == null || row.length != dimension)
				throw new IllegalArgumentException("data must be rectangular");
			for (double value : row) {
				if (!Double.isFinite(value) || requireUnit && !(value > 0.0 && value < 1.0))
					throw new IllegalArgumentException(requireUnit
							? "pseudo-observations must be finite and in (0, 1)"
							: "observations must be finite");
			}
		}
		return dimension;
	}

	private interface Objective { double value(double parameter); }
	private interface CopulaFactory { Copula create(double parameter); }
}
