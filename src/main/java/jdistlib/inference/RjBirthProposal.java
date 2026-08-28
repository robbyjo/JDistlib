/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Auxiliary-variable proposal used to create a newly active scalar parameter. */
public interface RjBirthProposal {
	double sample(int candidate, ReversibleJumpState state, RandomEngine random);
	double logDensity(double value, int candidate, ReversibleJumpState conditioningState);
	default void warmupUpdate(int candidate, double value, ReversibleJumpState conditioningState, boolean accepted) {}
	default Map<String, double[]> adaptationState() { return Collections.emptyMap(); }
	default void restoreAdaptation(Map<String, double[]> state) {
		if (state == null) throw new IllegalArgumentException("adaptation state required");
	}
	default void freezeAdaptation() {}
	default void resetAdaptation() {}
}
