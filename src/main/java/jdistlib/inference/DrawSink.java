/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Streaming destination for retained draws; implementations must copy if needed later. */
public interface DrawSink {
	void accept(int retainedIndex, double[] state, double logDensity, IterationStats statistics);
}
