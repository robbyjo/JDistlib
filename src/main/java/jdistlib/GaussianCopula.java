/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Gaussian copula parameterized by a positive-definite correlation matrix. */
public final class GaussianCopula implements Copula {
	private final double[][] correlation;
	private final double[] zero;
	private final MultivariateDistributionUtil.Factor factor;

	public GaussianCopula(double[][] correlation) {
		if (correlation == null) throw new IllegalArgumentException("correlation must not be null");
		CopulaUtil.requireMultivariateDimension(correlation.length);
		CopulaUtil.validateCorrelation(correlation);
		this.correlation = CopulaUtil.copyMatrix(correlation);
		this.zero = CopulaUtil.zeros(correlation.length);
		this.factor = MultivariateDistributionUtil.factor(this.correlation);
	}

	/** Constructs the Gaussian copula whose pairwise Kendall tau matrix is supplied. */
	public static GaussianCopula fromKendallsTau(double[][] tau) {
		return new GaussianCopula(correlationFromKendallsTau(tau));
	}

	/** Converts a Kendall tau matrix to its elliptical correlation matrix. */
	public static double[][] correlationFromKendallsTau(double[][] tau) {
		if (tau == null || tau.length < 2) {
			throw new IllegalArgumentException("tau matrix dimension must be at least two");
		}
		double[][] result = new double[tau.length][tau.length];
		for (int i = 0; i < tau.length; i++) {
			if (tau[i] == null || tau[i].length != tau.length)
				throw new IllegalArgumentException("tau must be a square matrix");
			for (int j = 0; j < tau.length; j++) {
				double value = tau[i][j];
				if (!Double.isFinite(value) || value < -1.0 || value > 1.0)
					throw new IllegalArgumentException("tau entries must be in [-1, 1]");
				if (i == j && value != 1.0)
					throw new IllegalArgumentException("tau diagonal must equal one");
				result[i][j] = i == j ? 1.0 : Math.sin(0.5 * Math.PI * value);
			}
		}
		return result;
	}

	@Override public int dimension() { return correlation.length; }

	public double[][] getCorrelation() { return CopulaUtil.copyMatrix(correlation); }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension())) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		double[] upper = new double[dimension()];
		for (int i = 0; i < upper.length; i++) {
			upper[i] = u[i] == 1.0 ? Double.POSITIVE_INFINITY
					: Normal.quantile(u[i], 0.0, 1.0, true, false);
		}
		return MultivariateNormal.cumulative(upper, zero, correlation).probability;
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension())) return Double.NaN;
		double[] z = new double[dimension()];
		double independentQuadratic = 0.0;
		for (int i = 0; i < z.length; i++) {
			z[i] = Normal.quantile(u[i], 0.0, 1.0, true, false);
			independentQuadratic += z[i] * z[i];
		}
		double correlatedQuadratic =
				MultivariateDistributionUtil.quadratic(z, zero, factor);
		return -0.5 * factor.logDeterminant
				- 0.5 * (correlatedQuadratic - independentQuadratic);
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[] z = MultivariateDistributionUtil.transform(zero, factor,
				MultivariateDistributionUtil.standardNormal(dimension(), random), 1.0);
		for (int i = 0; i < z.length; i++) z[i] = CopulaUtil.clampOpen(
				Normal.cumulative(z[i], 0.0, 1.0, true, false));
		return z;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension());
		return first == second ? 1.0
				: 2.0 / Math.PI * Math.asin(correlation[first][second]);
	}
}
