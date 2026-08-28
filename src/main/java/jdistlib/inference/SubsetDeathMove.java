/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Drops one uniformly selected active candidate with the matching reverse birth density. */
public final class SubsetDeathMove implements ReversibleJumpMove {
	private final RjBirthProposal birth;
	public SubsetDeathMove(RjBirthProposal birth) { if (birth == null) throw new IllegalArgumentException("birth proposal required"); this.birth = birth; }
	@Override public String name() { return "drop"; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) { require(target); return state != null && state.modelId() != 0L; }
	@Override public ReversibleJumpProposal propose(ReversibleJumpState state, ReversibleJumpTarget target, RandomEngine random) {
		SubsetSelectionTarget subset = require(target); int activeCount = Long.bitCount(state.modelId());
		if (activeCount == 0) return ReversibleJumpProposal.invalid("empty subset has no death move");
		int[] active = subset.activeCandidates(state.modelId()); int candidate = active[random.nextInt(active.length)];
		int position = subset.parameterIndex(state.modelId(), candidate); double value = state.parameter(position);
		long proposedModel = state.modelId() & ~(1L << candidate);
		CoordinateInsertionTransformation transform = new CoordinateInsertionTransformation(proposedModel, state.modelId(), position);
		DimensionMatchingResult mapped = transform.inverse(state, new double[0]);
		double forward = -Math.log(activeCount);
		double reverse = -Math.log(subset.candidateCount() - Long.bitCount(proposedModel))
				+ birth.logDensity(value, candidate, mapped.state());
		return ReversibleJumpProposal.valid(mapped.state(), "add", forward, reverse,
				transform.logAbsJacobian(state, new double[0], false));
	}
	@Override public void warmupUpdate(ReversibleJumpState current, ReversibleJumpProposal proposal,
			ReversibleJumpTarget target, boolean accepted) {
		if (!proposal.valid()) return; SubsetSelectionTarget subset = require(target);
		long difference = current.modelId() & ~proposal.proposedState().modelId(); int candidate = Long.numberOfTrailingZeros(difference);
		birth.warmupUpdate(candidate, current.parameter(subset.parameterIndex(current.modelId(), candidate)), proposal.proposedState(), accepted);
	}
	@Override public Map<String, double[]> adaptationState() { return birth.adaptationState(); }
	@Override public void restoreAdaptation(Map<String, double[]> state) { birth.restoreAdaptation(state); }
	@Override public void freezeAdaptation() { birth.freezeAdaptation(); }
	@Override public void resetAdaptation() { birth.resetAdaptation(); }
	private static SubsetSelectionTarget require(ReversibleJumpTarget target) {
		if (!(target instanceof SubsetSelectionTarget)) throw new IllegalArgumentException("subset move requires SubsetSelectionTarget"); return (SubsetSelectionTarget) target;
	}
}
