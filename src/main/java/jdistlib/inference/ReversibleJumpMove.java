/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** One dimension-changing or structure-changing reversible proposal. */
public interface ReversibleJumpMove {
	String name();
	boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target);
	ReversibleJumpProposal propose(ReversibleJumpState state,
			ReversibleJumpTarget target, RandomEngine random);
	default void warmupUpdate(ReversibleJumpState current, ReversibleJumpProposal proposal,
			ReversibleJumpTarget target, boolean accepted) {}
	default Map<String, double[]> adaptationState() { return Collections.emptyMap(); }
	default void restoreAdaptation(Map<String, double[]> state) {
		if (state == null) throw new IllegalArgumentException("adaptation state required");
	}
	default void freezeAdaptation() {}
	default void resetAdaptation() {}
}
