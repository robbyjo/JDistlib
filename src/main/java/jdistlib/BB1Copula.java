/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Bivariate BB1 (Clayton-Gumbel) copula, theta >= 0 and delta >= 1. */
public final class BB1Copula implements Copula {
	private final double theta, delta;
	public BB1Copula(double theta, double delta) {
		if (!(theta >= 0.0) || !(delta >= 1.0) || !Double.isFinite(theta) || !Double.isFinite(delta))
			throw new IllegalArgumentException("BB1 requires finite theta >= 0 and delta >= 1");
		this.theta = theta; this.delta = delta;
	}
	public double getTheta() { return theta; }
	public double getDelta() { return delta; }
	@Override public int dimension() { return 2; }
	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, 2)) return Double.NaN;
		if (u[0] == 0.0 || u[1] == 0.0) return 0.0;
		if (theta == 0.0) {
			return new GumbelCopula(2, delta).cumulative(u);
		}
		return Math.exp(-CopulaUtil.softplus(logRadialSum(u[0], u[1])) / theta);
	}
	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, 2)) return Double.NaN;
		if (theta == 0.0) return new GumbelCopula(2, delta).logDensity(u);
		if (delta == 1.0) return new ClaytonCopula(2, theta).logDensity(u);
		double logU = Math.log(u[0]), logV = Math.log(u[1]);
		double a = CopulaUtil.logExpm1(-theta * logU);
		double b = CopulaUtil.logExpm1(-theta * logV);
		double s = CopulaUtil.logAdd(delta * a, delta * b) / delta;
		double correction = CopulaUtil.logAdd(Math.log1p(theta),
				Math.log(theta) + Math.log(delta - 1.0) + CopulaUtil.softplus(-s));
		return -(1.0 + theta) * (logU + logV) + (delta - 1.0) * (a + b - 2.0 * s)
				- (1.0 / theta + 2.0) * CopulaUtil.softplus(s) + correction;
	}

	double conditionalSecond(double first, double second) {
		if (delta == 1.0) return new PairCopula(new ClaytonCopula(2,theta))
				.conditionalSecondGivenFirst(first,second);
		if (theta == 0.0) return new PairCopula(new GumbelCopula(2, delta))
				.conditionalSecondGivenFirst(first, second);
		if (first == 0.0) return 1.0;
		if (first == 1.0 && delta > 1.0) return 0.0;
		double s = logRadialSum(first, second);
		return Math.exp(-(1.0 + theta) * Math.log(first)
				+ (delta - 1.0) * (CopulaUtil.logExpm1(-theta * Math.log(first)) - s)
				- (1.0 / theta + 1.0) * CopulaUtil.softplus(s));
	}

	private double logRadialSum(double first, double second) {
		return CopulaUtil.logAdd(delta * CopulaUtil.logExpm1(-theta * Math.log(first)),
				delta * CopulaUtil.logExpm1(-theta * Math.log(second))) / delta;
	}
	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine required");
		double first = CopulaUtil.uniformOpen(random);
		return new double[] {first, new PairCopula(this).inverseSecondGivenFirst(first, CopulaUtil.uniformOpen(random))};
	}
	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, 2);
		return first == second ? 1.0 : 1.0 - 2.0 / (delta * (theta + 2.0));
	}
}
