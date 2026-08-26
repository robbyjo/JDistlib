/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Common contract for pair-copula vine constructions. */
public interface VineCopula extends Copula {
	/** Estimates the lower-orthant CDF and reports Monte Carlo uncertainty. */
	VineProbabilityResult cumulativeResult(double[] u, int samples,
			RandomEngine random);
}
