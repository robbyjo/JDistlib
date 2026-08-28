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
		double a = Math.pow(1.0 - u[0], theta);
		double b = Math.pow(1.0 - u[1], theta);
		return 1.0 - Math.pow(a + b - a * b, 1.0 / theta);
	}
	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, 2)) return Double.NaN;
		return Math.log(numericalDensity(u[0], u[1]));
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
	private double numericalDensity(double u, double v) {
		double h = 2e-5;
		double loU = Math.max(0.0, u - h), hiU = Math.min(1.0, u + h);
		double loV = Math.max(0.0, v - h), hiV = Math.min(1.0, v + h);
		double mixed = cumulative(new double[] {hiU, hiV}) - cumulative(new double[] {hiU, loV})
				- cumulative(new double[] {loU, hiV}) + cumulative(new double[] {loU, loV});
		return Math.max(Double.MIN_NORMAL, mixed / ((hiU - loU) * (hiV - loV)));
	}
	private double archimedeanTau() {
		int panels = 8192; double sum = 0.0;
		for (int i = 0; i < panels; i++) {
			double t = (i + 0.5) / panels;
			double power = Math.pow(1.0 - t, theta);
			double generator = -Math.log1p(-power);
			double derivative = -theta * Math.pow(1.0 - t, theta - 1.0) / (1.0 - power);
			sum += generator / derivative;
		}
		return 1.0 + 4.0 * sum / panels;
	}
}
