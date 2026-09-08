/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Bivariate Joe copula with upper-tail dependence and theta >= 1. */
public final class JoeCopula implements Copula {
	private final double theta;
	public JoeCopula(double theta) {
		if (!(theta >= 1.0) || !Double.isFinite(theta))
			throw new IllegalArgumentException("Joe theta must be finite and at least one");
		this.theta = theta;
	}
	public double getTheta() { return theta; }
	@Override public int dimension() { return 2; }
	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, 2)) return Double.NaN;
		if (theta == 1.0) return u[0] * u[1];
		if (CopulaUtil.hasZero(u)) return 0.0;
		return -Math.expm1(logGeneratorSum(u[0], u[1]) / theta);
	}
	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, 2)) return Double.NaN;
		if (theta == 1.0) return 0.0;
		double a = Math.log1p(-u[0]), b = Math.log1p(-u[1]);
		double sum = logGeneratorSum(u[0], u[1]);
		double correction = Math.log1p(-1.0 / theta)
				+ DistributionUtil.logOneMinusExp(theta * a)
				+ DistributionUtil.logOneMinusExp(theta * b);
		return Math.log(theta) + (theta - 1.0) * (a + b)
				+ (1.0 / theta - 2.0) * sum + CopulaUtil.logAdd(sum, correction);
	}
	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine required");
		double first = CopulaUtil.uniformOpen(random);
		double second = new PairCopula(this).inverseSecondGivenFirst(first, CopulaUtil.uniformOpen(random));
		return new double[] {first, second};
	}
	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, 2);
		return first == second ? 1.0 : archimedeanTau();
	}
	double conditionalSecond(double first, double second) {
		if (theta == 1.0) return second;
		if (first == 1.0) return 0.0;
		return Math.exp((theta - 1.0) * Math.log1p(-first)
				+ DistributionUtil.logOneMinusExp(theta * Math.log1p(-second))
				+ (1.0 / theta - 1.0) * logGeneratorSum(first, second));
	}

	private double logGeneratorSum(double first, double second) {
		double a = theta * Math.log1p(-first), b = theta * Math.log1p(-second);
		return CopulaUtil.logAdd(a, b + DistributionUtil.logOneMinusExp(a));
	}
	private double archimedeanTau() {
		if (theta == 1.0) return 0.0;
		double h = 2.0 / theta - 1.0;
		// The digamma quotient has a removable singularity at theta = 2.
		if (Math.abs(h) < 1e-3) {
			double quotient = 0.0, power = 1.0, factorial = 1.0;
			for (int order = 1; order <= 5; order++) {
				factorial *= order;
				quotient += jdistlib.math.PolyGamma.psigamma(2.0, order) * power / factorial;
				power *= h;
			}
			return 1.0 - 2.0 / theta * quotient;
		}
		return 1.0 + 2.0 / (2.0 - theta)
				* (jdistlib.math.PolyGamma.digamma(2.0)
				- jdistlib.math.PolyGamma.digamma(1.0 + 2.0 / theta));
	}
}
