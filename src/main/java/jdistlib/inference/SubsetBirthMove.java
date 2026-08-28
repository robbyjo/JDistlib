/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Adds one uniformly selected inactive candidate using a declared birth proposal. */
public final class SubsetBirthMove implements ReversibleJumpMove {
	private final RjBirthProposal birth;
	public SubsetBirthMove(RjBirthProposal birth) { if (birth == null) throw new IllegalArgumentException("birth proposal required"); this.birth = birth; }
	@Override public String name() { return "add"; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) {
		SubsetSelectionTarget subset = require(target); return state != null && Long.bitCount(state.modelId()) < subset.candidateCount();
	}
	@Override public ReversibleJumpProposal propose(ReversibleJumpState state, ReversibleJumpTarget target, RandomEngine random) {
		SubsetSelectionTarget subset = require(target); int inactive = subset.candidateCount() - Long.bitCount(state.modelId());
		if (inactive <= 0) return ReversibleJumpProposal.invalid("full subset has no birth move");
		int selected = random.nextInt(inactive), candidate = -1;
		for (int i = 0; i < subset.candidateCount(); i++) if (!subset.active(state.modelId(), i) && selected-- == 0) { candidate = i; break; }
		double value = birth.sample(candidate, state, random); long proposedModel = state.modelId() | 1L << candidate;
		int insertion = subset.commonDimension() + Long.bitCount(state.modelId() & ((1L << candidate) - 1L));
		CoordinateInsertionTransformation transform = new CoordinateInsertionTransformation(state.modelId(), proposedModel, insertion);
		DimensionMatchingResult mapped = transform.forward(state, new double[] {value});
		double forward = -Math.log(inactive) + birth.logDensity(value, candidate, state);
		double reverse = -Math.log(Long.bitCount(proposedModel));
		return ReversibleJumpProposal.valid(mapped.state(), "drop", forward, reverse,
				transform.logAbsJacobian(state, new double[] {value}, true));
	}
	@Override public void warmupUpdate(ReversibleJumpState current, ReversibleJumpProposal proposal,
			ReversibleJumpTarget target, boolean accepted) {
		if (!proposal.valid()) return; SubsetSelectionTarget subset = require(target);
		long difference = proposal.proposedState().modelId() & ~current.modelId(); int candidate = Long.numberOfTrailingZeros(difference);
		birth.warmupUpdate(candidate, proposal.proposedState().parameter(subset.parameterIndex(proposal.proposedState().modelId(), candidate)), current, accepted);
	}
	@Override public Map<String, double[]> adaptationState() { return birth.adaptationState(); }
	@Override public void restoreAdaptation(Map<String, double[]> state) { birth.restoreAdaptation(state); }
	@Override public void freezeAdaptation() { birth.freezeAdaptation(); }
	@Override public void resetAdaptation() { birth.resetAdaptation(); }
	private static SubsetSelectionTarget require(ReversibleJumpTarget target) {
		if (!(target instanceof SubsetSelectionTarget)) throw new IllegalArgumentException("subset move requires SubsetSelectionTarget"); return (SubsetSelectionTarget) target;
	}
}
