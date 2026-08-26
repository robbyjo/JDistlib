/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/**
 * Exchangeable Frank copula. Negative dependence is supported in two
 * dimensions; higher-dimensional instances require nonnegative dependence.
 */
public final class FrankCopula implements Copula {
	private final int dimension;
	private final double theta;
	private final double[] eulerian;

	public FrankCopula(int dimension, double theta) {
		CopulaUtil.requireMultivariateDimension(dimension);
		if (!Double.isFinite(theta))
			throw new IllegalArgumentException("Frank theta must be finite");
		if (dimension > 2 && theta < 0.0) {
			throw new IllegalArgumentException(
					"negative Frank dependence is only valid in two dimensions");
		}
		this.dimension = dimension;
		this.theta = theta;
		this.eulerian = eulerianNumbers(dimension - 1);
	}

	public static FrankCopula fromKendallsTau(int dimension, double tau) {
		return new FrankCopula(dimension, parameterFromKendallsTau(tau));
	}

	/** Numerically inverts the Frank tau relationship. */
	public static double parameterFromKendallsTau(double tau) {
		if (!Double.isFinite(tau) || !(tau > -1.0 && tau < 1.0))
			throw new IllegalArgumentException("Frank tau must be in (-1, 1)");
		if (tau == 0.0) return 0.0;
		double target = Math.abs(tau);
		double low = 0.0;
		double high = 2.0;
		while (tauForPositiveTheta(high) < target && high < 1e12) high *= 2.0;
		for (int i = 0; i < 100; i++) {
			double middle = low + (high - low) / 2.0;
			if (tauForPositiveTheta(middle) < target) low = middle;
			else high = middle;
		}
		double result = low + (high - low) / 2.0;
		return tau < 0.0 ? -result : result;
	}

	@Override public int dimension() { return dimension; }
	public double getTheta() { return theta; }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension)) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		if (theta == 0.0) return product(u);
		if (theta < 0.0) {
			return u[0] - positiveCumulative(-theta, u[0], 1.0 - u[1]);
		}
		double logP = logOneMinusExponential(theta);
		double logZ = logP;
		for (double value : u) {
			logZ += logOneMinusExponential(theta * value) - logP;
		}
		return -DistributionUtil.logOneMinusExp(logZ) / theta;
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension)) return Double.NaN;
		if (theta == 0.0) return 0.0;
		if (theta < 0.0) return positiveLogDensity(-theta, u[0], 1.0 - u[1]);
		double logP = logOneMinusExponential(theta);
		double logZ = logP;
		double inverseDerivativeLog = 0.0;
		for (double value : u) {
			double logA = logOneMinusExponential(theta * value);
			logZ += logA - logP;
			inverseDerivativeLog += Math.log(theta) - theta * value - logA;
		}
		double z = Math.exp(logZ);
		double logPoly = logEulerianPolynomial(z);
		double logPolylogarithm = logZ + logPoly
				- dimension * DistributionUtil.logOneMinusExp(logZ);
		return -Math.log(theta) + logPolylogarithm + inverseDerivativeLog;
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		if (theta == 0.0) return new IndependenceCopula(dimension).random(random);
		if (theta < 0.0) {
			double[] positive = positiveRandom(2, -theta, random);
			positive[1] = CopulaUtil.clampOpen(1.0 - positive[1]);
			return positive;
		}
		return positiveRandom(dimension, theta, random);
	}

	private static double[] positiveRandom(int dimension, double parameter,
			RandomEngine random) {
		double logP = logOneMinusExponential(parameter);
		double frailty = logarithmicRandom(logP, parameter, random);
		double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			double exponential = -Math.log(CopulaUtil.uniformOpen(random));
			double logZ = logP - exponential / frailty;
			result[i] = CopulaUtil.clampOpen(
					-DistributionUtil.logOneMinusExp(logZ) / parameter);
		}
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension);
		if (first == second) return 1.0;
		return theta < 0.0 ? -tauForPositiveTheta(-theta) : tauForPositiveTheta(theta);
	}

	private static double positiveCumulative(double parameter, double first,
			double second) {
		double logP = logOneMinusExponential(parameter);
		double logZ = logOneMinusExponential(parameter * first)
				+ logOneMinusExponential(parameter * second) - logP;
		return -DistributionUtil.logOneMinusExp(logZ) / parameter;
	}

	private static double positiveLogDensity(double parameter, double first,
			double second) {
		double logP = logOneMinusExponential(parameter);
		double logZ = logOneMinusExponential(parameter * first)
				+ logOneMinusExponential(parameter * second) - logP;
		return Math.log(parameter) - logP - parameter * (first + second)
				- 2.0 * DistributionUtil.logOneMinusExp(logZ);
	}

	private double logEulerianPolynomial(double z) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (int k = 0; k < eulerian.length; k++) {
			if (eulerian[k] > 0.0)
				maximum = Math.max(maximum, Math.log(eulerian[k]) + k * Math.log(z));
		}
		double sum = 0.0;
		for (int k = 0; k < eulerian.length; k++) {
			if (eulerian[k] > 0.0)
				sum += Math.exp(Math.log(eulerian[k]) + k * Math.log(z) - maximum);
		}
		return maximum + Math.log(sum);
	}

	private static double product(double[] values) {
		double result = 1.0;
		for (double value : values) result *= value;
		return result;
	}

	private static double logarithmicRandom(double logP, double parameter,
			RandomEngine random) {
		double second = CopulaUtil.uniformOpen(random);
		double logSecond = Math.log(second);
		if (logSecond > logP) return 1.0;
		double first = CopulaUtil.uniformOpen(random);
		double logQ = logOneMinusExponential(first * parameter);
		if (logSecond <= 2.0 * logQ)
			return Math.floor(1.0 + logSecond / logQ);
		return logSecond > logQ ? 1.0 : 2.0;
	}

	private static double logOneMinusExponential(double positiveValue) {
		return DistributionUtil.logOneMinusExp(-positiveValue);
	}

	private static double tauForPositiveTheta(double value) {
		if (value == 0.0) return 0.0;
		if (value < 1e-3) {
			double square = value * value;
			return value * (1.0 / 9.0 + square * (-1.0 / 900.0
					+ square / 52920.0));
		}
		double integral;
		if (value > 50.0) {
			integral = Math.PI * Math.PI / 6.0;
		} else {
			int intervals = 4096;
			double width = value / intervals;
			double sum = 1.0 + debyeIntegrand(value);
			for (int i = 1; i < intervals; i++) {
				sum += (i % 2 == 0 ? 2.0 : 4.0) * debyeIntegrand(i * width);
			}
			integral = sum * width / 3.0;
		}
		return 1.0 - 4.0 / value + 4.0 * integral / (value * value);
	}

	private static double debyeIntegrand(double value) {
		return value == 0.0 ? 1.0 : value / Math.expm1(value);
	}

	private static double[] eulerianNumbers(int order) {
		double[] current = new double[] {1.0};
		for (int n = 2; n <= order; n++) {
			double[] next = new double[n];
			for (int k = 0; k < n; k++) {
				double left = k == 0 ? 0.0 : (n - k) * current[k - 1];
				double right = k >= current.length ? 0.0 : (k + 1.0) * current[k];
				next[k] = left + right;
			}
			current = next;
		}
		return current;
	}
}
