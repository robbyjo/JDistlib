/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Common contract for a reproducible MCMC chain. */
public interface Sampler {
	ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random);
}
