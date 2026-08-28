/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Normalized proposal over candidates inactive in the conditioning model. */
public interface SparseCandidateProposal extends AutoCloseable {
	SparseCandidateChoice sample(SparseSubsetState state, SparseSubsetTarget target, RandomEngine random);
	double logProbability(int candidate, SparseSubsetState state, SparseSubsetTarget target);
	@Override default void close() {}
}
