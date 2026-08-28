/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** One support-aware transition in a scheduled hybrid sampler. */
public interface HybridKernel {
	String name();
	HybridKernelTransition update(double[] state, double currentLogDensity,
			LogDensity target, MixedStateSpace space, RandomEngine random);
}
