/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Product copula representing mutual independence. */
public final class IndependenceCopula implements Copula {
	private final int dimension;

	public IndependenceCopula(int dimension) {
		CopulaUtil.requireDimension(dimension);
		this.dimension = dimension;
	}

	@Override public int dimension() { return dimension; }

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension)) return Double.NaN;
		double result = 1.0;
		for (double value : u) result *= value;
		return result;
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension)) return Double.NaN;
		return 0.0;
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) result[i] = CopulaUtil.uniformOpen(random);
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension);
		return first == second ? 1.0 : 0.0;
	}

	@Override public CopulaDiagnostics diagnose(double[] u) {
		return CopulaDiagnostics.inspect(u, dimension, true);
	}
}
