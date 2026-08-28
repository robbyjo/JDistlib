/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Dimension-matching proposal for a coefficient born into a sparse model. */
public interface SparseCoefficientProposal {
	double sample(int candidate, SparseSubsetState conditioningState, RandomEngine random);
	double logDensity(double value, int candidate, SparseSubsetState conditioningState);
}
