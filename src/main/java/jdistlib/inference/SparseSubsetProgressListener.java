/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Progress callback for one restartable sparse RJMCMC segment. */
@FunctionalInterface
public interface SparseSubsetProgressListener {
	void update(int segmentCompleted, int segmentTotal, long totalCompleted, boolean warmup,
			SparseSubsetIterationStats statistics);
}
