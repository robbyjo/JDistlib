/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Lightweight callback invoked after a completed sampler transition. */
public interface ProgressListener {
	void update(int completed, int total, boolean warmup, IterationStats statistics);
}
