/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Student-t copula parameterized by correlation and degrees of freedom. */
public final class StudentTCopula implements Copula {
	private final double[][] correlation;
	private final double degreesOfFreedom;
	private final double[] zero;
	private final MultivariateDistributionUtil.Factor factor;

	public StudentTCopula(double[][] correlation, double degreesOfFreedom) {
		if (correlation == null) throw new IllegalArgumentException("correlation must not be null");
		CopulaUtil.requireMultivariateDimension(correlation.length);
		CopulaUtil.validateCorrelation(correlation);
		if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom))
			throw new IllegalArgumentException("degrees of freedom must be finite and positive");
		this.correlation = CopulaUtil.copyMatrix(correlation);
		this.degreesOfFreedom = degreesOfFreedom;
		this.zero = CopulaUtil.zeros(correlation.length);
		this.factor = MultivariateDistributionUtil.factor(this.correlation);
	}

	public static StudentTCopula fromKendallsTau(double[][] tau,
			double degreesOfFreedom) {
		return new StudentTCopula(GaussianCopula.correlationFromKendallsTau(tau),
				degreesOfFreedom);
	}

	@Override public int dimension() { return correlation.length; }
	public double[][] getCorrelation() { return CopulaUtil.copyMatrix(correlation); }
	public double getDegreesOfFreedom() { return degreesOfFreedom; }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension())) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		double[] upper = new double[dimension()];
		for (int i = 0; i < upper.length; i++) {
			upper[i] = u[i] == 1.0 ? Double.POSITIVE_INFINITY
					: T.quantile(u[i], degreesOfFreedom, true, false);
		}
		return MultivariateStudentT.cumulative(upper, zero, correlation,
				degreesOfFreedom).probability;
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension())) return Double.NaN;
		double[] x = new double[dimension()];
		double marginalLogDensity = 0.0;
		for (int i = 0; i < x.length; i++) {
			x[i] = T.quantile(u[i], degreesOfFreedom, true, false);
			marginalLogDensity += T.density(x[i], degreesOfFreedom, true);
		}
		double quadratic = MultivariateDistributionUtil.quadratic(x, zero, factor);
		int d = dimension();
		double jointLogDensity = lgammafn((degreesOfFreedom + d) / 2.0)
				- lgammafn(degreesOfFreedom / 2.0)
				- 0.5 * (d * Math.log(degreesOfFreedom * Math.PI)
				+ factor.logDeterminant)
				- 0.5 * (degreesOfFreedom + d)
				* Math.log1p(quadratic / degreesOfFreedom);
		return jointLogDensity - marginalLogDensity;
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double multiplier = Math.sqrt(degreesOfFreedom
				/ ChiSquare.random(degreesOfFreedom, random));
		double[] x = MultivariateDistributionUtil.transform(zero, factor,
				MultivariateDistributionUtil.standardNormal(dimension(), random), multiplier);
		for (int i = 0; i < x.length; i++) {
			x[i] = CopulaUtil.clampOpen(
					T.cumulative(x[i], degreesOfFreedom, true, false));
		}
		return x;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension());
		return first == second ? 1.0
				: 2.0 / Math.PI * Math.asin(correlation[first][second]);
	}
}
