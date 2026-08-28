/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Gaussian prior/reference measure used by elliptical slice and pCN updates. */
public interface GaussianReference {
	double[] mean();
	double[] random(RandomEngine random);
	/** Independent Gaussian reference parameterized by marginal standard deviations. */
	static GaussianReference diagonal(final double[] mean, final double[] standardDeviation) {
		if (mean == null || standardDeviation == null || mean.length == 0
				|| mean.length != standardDeviation.length)
			throw new IllegalArgumentException("matching nonempty Gaussian parameters are required");
		final double[] center = mean.clone(), scale = standardDeviation.clone();
		for (double value : scale) if (!(value > 0.0) || !Double.isFinite(value))
			throw new IllegalArgumentException("standard deviations must be finite and positive");
		return new GaussianReference() {
			@Override public double[] mean() { return center.clone(); }
			@Override public double[] random(RandomEngine random) {
				double[] result = center.clone();
				for (int i = 0; i < result.length; i++) result[i] += scale[i] * random.nextGaussian();
				return result;
			}
		};
	}
}
