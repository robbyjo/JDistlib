/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Progress callback for one trans-dimensional chain. */
@FunctionalInterface
public interface ReversibleJumpProgressListener {
	void update(int completed, int total, boolean warmup, ReversibleJumpIterationStats statistics);
}
