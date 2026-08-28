/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Streaming sink for ragged retained RJ draws. */
@FunctionalInterface
public interface ReversibleJumpDrawSink {
	void accept(int retained, ReversibleJumpState state, double logJoint, ReversibleJumpIterationStats statistics);
}
