/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Exchanges one active and inactive candidate while preserving model dimension. */
public final class SubsetSwapMove implements ReversibleJumpMove {
	private final RjBirthProposal birth;
	public SubsetSwapMove(RjBirthProposal birth) { if (birth == null) throw new IllegalArgumentException("birth proposal required"); this.birth = birth; }
	@Override public String name() { return "swap"; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) {
		SubsetSelectionTarget subset = require(target); int active = state == null ? 0 : Long.bitCount(state.modelId()); return active > 0 && active < subset.candidateCount();
	}
	@Override public ReversibleJumpProposal propose(ReversibleJumpState state, ReversibleJumpTarget target, RandomEngine random) {
		SubsetSelectionTarget subset = require(target); int[] active = subset.activeCandidates(state.modelId()); int removed = active[random.nextInt(active.length)];
		int inactiveChoice = random.nextInt(subset.candidateCount() - active.length), added = -1;
		for (int candidate = 0; candidate < subset.candidateCount(); candidate++) if (!subset.active(state.modelId(), candidate) && inactiveChoice-- == 0) { added = candidate; break; }
		double removedValue = state.parameter(subset.parameterIndex(state.modelId(), removed)); double addedValue = birth.sample(added, state, random);
		long proposedModel = state.modelId() & ~(1L << removed) | 1L << added; int[] proposedActive = subset.activeCandidates(proposedModel);
		double[] old = state.parameters(), proposed = new double[old.length]; System.arraycopy(old, 0, proposed, 0, subset.commonDimension());
		for (int i = 0; i < proposedActive.length; i++) {
			int candidate = proposedActive[i]; proposed[subset.commonDimension() + i] = candidate == added
					? addedValue : state.parameter(subset.parameterIndex(state.modelId(), candidate));
		}
		ReversibleJumpState result = new ReversibleJumpState(proposedModel, proposed);
		double choices = -Math.log(active.length) - Math.log(subset.candidateCount() - active.length);
		return ReversibleJumpProposal.valid(result, "swap", choices + birth.logDensity(addedValue, added, state),
				choices + birth.logDensity(removedValue, removed, result), 0.0);
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
