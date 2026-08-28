/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** A fixed-model update scheduled between trans-dimensional proposals. */
public interface ReversibleJumpWithinModelKernel {
	String name();
	boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target);
	ReversibleJumpWithinModelTransition update(ReversibleJumpState state, double currentLogJoint,
			ReversibleJumpTarget target, RandomEngine random, boolean warmup);
	default Map<String, double[]> adaptationState() { return Collections.emptyMap(); }
	default void restoreAdaptation(Map<String, double[]> state) {
		if (state == null) throw new IllegalArgumentException("adaptation state required");
	}
	default void freezeAdaptation() {}
	default void resetAdaptation() {}
}
