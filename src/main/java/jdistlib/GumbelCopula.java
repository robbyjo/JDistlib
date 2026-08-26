/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Exchangeable Gumbel copula. */
public final class GumbelCopula implements Copula {
	private final int dimension;
	private final double theta;
	private final double alpha;
	private final double[] derivativePolynomial;

	public GumbelCopula(int dimension, double theta) {
		CopulaUtil.requireMultivariateDimension(dimension);
		if (!(theta >= 1.0) || !Double.isFinite(theta))
			throw new IllegalArgumentException("Gumbel theta must be finite and at least one");
		this.dimension = dimension;
		this.theta = theta;
		this.alpha = 1.0 / theta;
		this.derivativePolynomial = derivativePolynomial(dimension, alpha);
	}

	public static GumbelCopula fromKendallsTau(int dimension, double tau) {
		return new GumbelCopula(dimension, parameterFromKendallsTau(tau));
	}

	public static double parameterFromKendallsTau(double tau) {
		if (!(tau >= 0.0 && tau < 1.0) || !Double.isFinite(tau))
			throw new IllegalArgumentException("Gumbel tau must be in [0, 1)");
		return 1.0 / (1.0 - tau);
	}

	@Override public int dimension() { return dimension; }
	public double getTheta() { return theta; }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension)) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		if (theta == 1.0) {
			double product = 1.0;
			for (double value : u) product *= value;
			return product;
		}
		double logT = logGeneratorSum(u);
		return Math.exp(-Math.exp(logT / theta));
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension)) return Double.NaN;
		if (theta == 1.0) return 0.0;
		double logT = logGeneratorSum(u);
		double x = Math.exp(alpha * logT);
		double sumLogNegativeLog = 0.0;
		double sumLogU = 0.0;
		for (double value : u) {
			sumLogNegativeLog += Math.log(-Math.log(value));
			sumLogU += Math.log(value);
		}
		return -x - dimension * logT + logPolynomial(x)
				+ dimension * Math.log(theta)
				+ (theta - 1.0) * sumLogNegativeLog - sumLogU;
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		if (theta == 1.0) return new IndependenceCopula(dimension).random(random);
		double angle = Math.PI * CopulaUtil.uniformOpen(random);
		double exponential = -Math.log(CopulaUtil.uniformOpen(random));
		double logFrailty = Math.log(Math.sin(alpha * angle))
				- Math.log(Math.sin(angle)) / alpha
				+ (1.0 - alpha) / alpha
				* (Math.log(Math.sin((1.0 - alpha) * angle))
				- Math.log(exponential));
		double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			double logExponential = Math.log(-Math.log(CopulaUtil.uniformOpen(random)));
			double exponentLog = alpha * (logExponential - logFrailty);
			result[i] = CopulaUtil.clampOpen(
					exponentLog > Math.log(Double.MAX_VALUE) ? 0.0
					: Math.exp(-Math.exp(exponentLog)));
		}
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension);
		return first == second ? 1.0 : 1.0 - 1.0 / theta;
	}

	private double logGeneratorSum(double[] u) {
		double[] terms = new double[dimension];
		for (int i = 0; i < dimension; i++) terms[i] = theta * Math.log(-Math.log(u[i]));
		return CopulaUtil.logSumExp(terms);
	}

	private double logPolynomial(double x) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (int k = 1; k < derivativePolynomial.length; k++) {
			if (derivativePolynomial[k] > 0.0) {
				maximum = Math.max(maximum,
						Math.log(derivativePolynomial[k]) + k * Math.log(x));
			}
		}
		double sum = 0.0;
		for (int k = 1; k < derivativePolynomial.length; k++) {
			if (derivativePolynomial[k] > 0.0) {
				sum += Math.exp(Math.log(derivativePolynomial[k])
						+ k * Math.log(x) - maximum);
			}
		}
		return maximum + Math.log(sum);
	}

	private static double[] derivativePolynomial(int order, double alpha) {
		double[] coefficients = new double[order + 1];
		coefficients[1] = alpha;
		for (int n = 1; n < order; n++) {
			double[] next = new double[order + 1];
			for (int k = 1; k <= n; k++) {
				next[k] += (n - alpha * k) * coefficients[k];
				next[k + 1] += alpha * coefficients[k];
			}
			coefficients = next;
		}
		return coefficients;
	}
}
