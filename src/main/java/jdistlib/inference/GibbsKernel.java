/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** One exact or MCMC-within-Gibbs state update. */
@FunctionalInterface
public interface GibbsKernel {
	void update(double[] state, LogDensity target, RandomEngine random);
}
