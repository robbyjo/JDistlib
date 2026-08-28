/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Reusable one-step Markov kernel used by chains and meta-samplers. */
public interface TransitionKernel<S> {
	S initialize(LogDensity target, double[] initialState, SamplingOptions options,
			RandomEngine random);
	KernelTransition<S> step(LogDensity target, S state, SamplingOptions options,
			RandomEngine random);
}
