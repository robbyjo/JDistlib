/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Bivariate 90, 180 (survival), or 270 degree rotation of a copula. */
public final class RotatedCopula implements Copula {
	public enum Rotation { CLOCKWISE_90, SURVIVAL_180, CLOCKWISE_270 }
	private final Copula base;
	private final Rotation rotation;
	public RotatedCopula(Copula base, Rotation rotation) {
		if (base == null || base.dimension() != 2 || rotation == null)
			throw new IllegalArgumentException("a bivariate base copula and rotation are required");
		this.base = base; this.rotation = rotation;
	}
	public Copula getBase() { return base; }
	public Rotation getRotation() { return rotation; }
	@Override public int dimension() { return 2; }
	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, 2)) return Double.NaN;
		if (rotation == Rotation.CLOCKWISE_90)
			return u[1] - base.cumulative(new double[] {1.0 - u[0], u[1]});
		if (rotation == Rotation.SURVIVAL_180)
			return u[0] + u[1] - 1.0 + base.cumulative(new double[] {1.0 - u[0], 1.0 - u[1]});
		return u[0] - base.cumulative(new double[] {u[0], 1.0 - u[1]});
	}
	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, 2)) return Double.NaN;
		if (rotation == Rotation.CLOCKWISE_90) return base.logDensity(new double[] {1.0 - u[0], u[1]});
		if (rotation == Rotation.SURVIVAL_180) return base.logDensity(new double[] {1.0 - u[0], 1.0 - u[1]});
		return base.logDensity(new double[] {u[0], 1.0 - u[1]});
	}
	@Override public double[] random(RandomEngine random) {
		double[] draw = base.random(random);
		if (rotation == Rotation.CLOCKWISE_90) draw[0] = 1.0 - draw[0];
		else if (rotation == Rotation.SURVIVAL_180) { draw[0] = 1.0 - draw[0]; draw[1] = 1.0 - draw[1]; }
		else draw[1] = 1.0 - draw[1];
		return draw;
	}
	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, 2);
		if (first == second) return 1.0;
		double tau = base.kendallsTau(first, second);
		return rotation == Rotation.SURVIVAL_180 ? tau : -tau;
	}
}
