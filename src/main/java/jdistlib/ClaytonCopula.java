/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Exchangeable Clayton copula with nonnegative dependence. */
public final class ClaytonCopula implements Copula {
	private final int dimension;
	private final double theta;

	public ClaytonCopula(int dimension, double theta) {
		CopulaUtil.requireMultivariateDimension(dimension);
		if (!(theta >= 0.0) || !Double.isFinite(theta)) {
			throw new IllegalArgumentException("Clayton theta must be finite and nonnegative");
		}
		this.dimension = dimension;
		this.theta = theta;
	}

	public static ClaytonCopula fromKendallsTau(int dimension, double tau) {
		return new ClaytonCopula(dimension, parameterFromKendallsTau(tau));
	}

	public static double parameterFromKendallsTau(double tau) {
		if (!(tau >= 0.0 && tau < 1.0) || !Double.isFinite(tau))
			throw new IllegalArgumentException("Clayton tau must be in [0, 1)");
		return 2.0 * tau / (1.0 - tau);
	}

	@Override public int dimension() { return dimension; }
	public double getTheta() { return theta; }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension)) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		if (theta == 0.0) {
			double product = 1.0;
			for (double value : u) product *= value;
			return product;
		}
		double excess = 0.0;
		for (double value : u) excess += Math.expm1(-theta * Math.log(value));
		return Math.exp(-Math.log1p(excess) / theta);
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension)) return Double.NaN;
		if (theta == 0.0) return 0.0;
		double sumLogU = 0.0;
		double excess = 0.0;
		for (int i = 0; i < dimension; i++) {
			sumLogU += Math.log(u[i]);
			excess += Math.expm1(-theta * Math.log(u[i]));
		}
		double logGeneratorSum = Math.log1p(excess);
		double logCoefficient = 0.0;
		for (int k = 1; k < dimension; k++) logCoefficient += Math.log1p(k * theta);
		return logCoefficient - (1.0 + theta) * sumLogU
				- (dimension + 1.0 / theta) * logGeneratorSum;
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		if (theta == 0.0) return new IndependenceCopula(dimension).random(random);
		double frailty = Gamma.random(1.0 / theta, 1.0, random);
		double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			double exponential = -Math.log(CopulaUtil.uniformOpen(random));
			result[i] = CopulaUtil.clampOpen(
					Math.exp(-Math.log1p(exponential / frailty) / theta));
		}
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension);
		return first == second ? 1.0 : theta / (theta + 2.0);
	}
}
