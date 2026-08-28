/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import java.util.Arrays;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Elliptical multivariate power-exponential (generalized Gaussian) law. */
public final class MultivariatePowerExponential {
	private static final long PROBABILITY_SEED = 0x4d76506f77457870L;
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

	/**
	 * Computes a rectangle probability by integrating the exact conditional
	 * radial probability over uniformly distributed directions. This radial
	 * conditioning remains valid for every positive shape, including the
	 * non-Gaussian cases that are not normal scale mixtures.
	 */
	public static MultivariateProbabilityResult probability(final double[] lower,
			final double[] upper, final double[] location, final double[][] scatter,
			final double shape, MultivariateProbabilityOptions options,
			RandomEngine random) {
		if (!(shape > 0.0) || !Double.isFinite(shape) || lower == null ||
				upper == null || location == null || lower.length != location.length ||
				upper.length != location.length)
			return invalidProbability();
		final MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(location, scatter);
		if (factor == null) return invalidProbability();
		boolean unrestricted = true;
		for (int i = 0; i < location.length; i++) {
			if (Double.isNaN(lower[i]) || Double.isNaN(upper[i]))
				return invalidProbability();
			if (!(lower[i] < upper[i])) return exactProbability(0.0);
			unrestricted &= lower[i] == Double.NEGATIVE_INFINITY &&
					upper[i] == Double.POSITIVE_INFINITY;
		}
		if (unrestricted) return exactProbability(1.0);
		boolean centralOrthant = true;
		for (int i = 0; i < location.length; i++)
			centralOrthant &= (lower[i] == Double.NEGATIVE_INFINITY &&
					upper[i] == location[i]) || (lower[i] == location[i] &&
					upper[i] == Double.POSITIVE_INFINITY);
		// Every centered elliptical law has the same angular distribution as MVN.
		if (shape == 1.0 || centralOrthant)
			return MultivariateNormal.probability(lower, upper, location, scatter,
					options, random);
		if (location.length == 1) {
			double scale = factor.lower[0][0];
			double positive = radialInterval(Math.max(0.0,
					(lower[0] - location[0]) / scale), Math.max(0.0,
					(upper[0] - location[0]) / scale), 1, shape);
			double negative = radialInterval(Math.max(0.0,
					(location[0] - upper[0]) / scale), Math.max(0.0,
					(location[0] - lower[0]) / scale), 1, shape);
			return exactProbability(0.5 * (positive + negative));
		}
		MultivariateProbabilityResult estimate = MultivariateProbability.integrate(
				new MultivariateProbability.Integrand() {
			@Override public int dimension() {
				return location.length == 2 ? 1 : location.length;
			}
			@Override public double value(double[] uniforms) {
				int dimension = location.length;
				double[] direction = new double[dimension];
				if (dimension == 2) {
					double angle = 2.0 * Math.PI * uniforms[0];
					direction[0] = Math.cos(angle);
					direction[1] = Math.sin(angle);
				} else {
					double norm = 0.0;
					for (int i = 0; i < dimension; i++) {
						double u = Math.max(Math.nextUp(0.0), Math.min(
								Math.nextAfter(1.0, 0.0), uniforms[i]));
						direction[i] = Normal.quantile(u, 0.0, 1.0, true, false);
						norm += direction[i] * direction[i];
					}
					if (!(norm > 0.0) || !Double.isFinite(norm)) return 0.0;
					norm = Math.sqrt(norm);
					for (int i = 0; i < dimension; i++) direction[i] /= norm;
				}
				double radialLower = 0.0;
				double radialUpper = Double.POSITIVE_INFINITY;
				for (int i = 0; i < dimension; i++) {
					double velocity = 0.0;
					for (int j = 0; j <= i; j++)
						velocity += factor.lower[i][j] * direction[j];
					if (velocity == 0.0) {
						if (location[i] < lower[i] || location[i] > upper[i]) return 0.0;
						continue;
					}
					double left = (lower[i] - location[i]) / velocity;
					double right = (upper[i] - location[i]) / velocity;
					if (left > right) { double swap = left; left = right; right = swap; }
					radialLower = Math.max(radialLower, left);
					radialUpper = Math.min(radialUpper, right);
					if (!(radialLower < radialUpper)) return 0.0;
				}
				return radialInterval(radialLower, radialUpper, dimension, shape);
			}
		}, options, random);
		// An all-zero randomized sample is not evidence of convergence in a tail.
		if (estimate.probability == 0.0 && estimate.evaluations > 0)
			return new MultivariateProbabilityResult(0.0,
					Math.max(estimate.absoluteError, 1.0 / estimate.evaluations),
					estimate.evaluations, 1);
		return estimate;
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] scatter, double shape) {
		return probability(lower, upper, location, scatter, shape,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P(X[i] <= upper[i], all i)}. */
	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scatter, double shape,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (upper == null) return invalidProbability();
		double[] lower = new double[upper.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		return probability(lower, upper, location, scatter, shape, options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scatter, double shape) {
		return cumulative(upper, location, scatter, shape,
				new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	private static double radialInterval(double lower, double upper, int dimension,
			double shape) {
		lower = Math.max(0.0, lower);
		if (!(lower < upper)) return 0.0;
		double gammaShape = dimension / (2.0 * shape);
		double lowerArgument = radialGammaArgument(lower, shape);
		double upperArgument = radialGammaArgument(upper, shape);
		if (lowerArgument >= gammaShape) {
			return Math.max(0.0, Gamma.cumulative(lowerArgument, gammaShape, 1.0,
					false, false) - Gamma.cumulative(upperArgument, gammaShape, 1.0,
					false, false));
		}
		return Math.max(0.0, Gamma.cumulative(upperArgument, gammaShape, 1.0,
				true, false) - Gamma.cumulative(lowerArgument, gammaShape, 1.0,
				true, false));
	}

	private static double radialGammaArgument(double radius, double shape) {
		if (Double.isInfinite(radius)) return Double.POSITIVE_INFINITY;
		if (!(radius > 0.0)) return 0.0;
		double logValue = 2.0 * shape * Math.log(radius) - Math.log(2.0);
		return logValue > Math.log(Double.MAX_VALUE) ? Double.POSITIVE_INFINITY :
				Math.exp(logValue);
	}

	private static MultivariateProbabilityResult exactProbability(double value) {
		return new MultivariateProbabilityResult(value, 0.0, 0, 0);
	}

	private static MultivariateProbabilityResult invalidProbability() {
		return new MultivariateProbabilityResult(Double.NaN, Double.NaN, 0, 2);
	}
}
