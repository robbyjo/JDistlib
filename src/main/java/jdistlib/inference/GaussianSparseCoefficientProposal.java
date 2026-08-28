/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Independent Gaussian sparse-coefficient birth proposal. */
public final class GaussianSparseCoefficientProposal implements SparseCoefficientProposal {
	private final double mean, standardDeviation;
	public GaussianSparseCoefficientProposal(double mean, double standardDeviation) {
		if (!Double.isFinite(mean) || !(standardDeviation > 0.0) || !Double.isFinite(standardDeviation)) throw new IllegalArgumentException("finite Gaussian proposal required");
		this.mean = mean; this.standardDeviation = standardDeviation;
	}
	@Override public double sample(int candidate, SparseSubsetState state, RandomEngine random) {
		if (candidate < 0 || state == null || random == null) throw new IllegalArgumentException("candidate, state, and random required");
		return mean + standardDeviation * random.nextGaussian();
	}
	@Override public double logDensity(double value, int candidate, SparseSubsetState state) {
		if (!Double.isFinite(value) || candidate < 0 || state == null) throw new IllegalArgumentException("finite coefficient and state required");
		double z = (value - mean) / standardDeviation;
		return -0.5 * z * z - Math.log(standardDeviation) - 0.5 * Math.log(2.0 * Math.PI);
	}
}
