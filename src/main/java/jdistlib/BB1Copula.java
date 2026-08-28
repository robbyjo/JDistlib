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
			double sum = Math.pow(-Math.log(u[0]), delta) + Math.pow(-Math.log(u[1]), delta);
			return Math.exp(-Math.pow(sum, 1.0 / delta));
		}
		double a = Math.pow(Math.expm1(-theta * Math.log(u[0])), delta);
		double b = Math.pow(Math.expm1(-theta * Math.log(u[1])), delta);
		return Math.exp(-Math.log1p(Math.pow(a + b, 1.0 / delta)) / theta);
	}
	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, 2)) return Double.NaN;
		double h = 2e-5, a = u[0], b = u[1];
		double loA = Math.max(0.0, a-h), hiA = Math.min(1.0, a+h);
		double loB = Math.max(0.0, b-h), hiB = Math.min(1.0, b+h);
		double mixed = cumulative(new double[] {hiA,hiB})-cumulative(new double[] {hiA,loB})
				-cumulative(new double[] {loA,hiB})+cumulative(new double[] {loA,loB});
		return Math.log(Math.max(Double.MIN_NORMAL, mixed / ((hiA-loA)*(hiB-loB))));
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
