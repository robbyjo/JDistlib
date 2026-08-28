/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Streaming callback for retained sparse draws. */
@FunctionalInterface
public interface SparseSubsetDrawSink {
	void accept(long retainedIndex, SparseSubsetState state, double logJoint,
			SparseSubsetIterationStats statistics);
}
