/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Uniform proposal over currently inactive sparse candidates. */
public final class UniformSparseCandidateProposal implements SparseCandidateProposal {
	@Override public SparseCandidateChoice sample(SparseSubsetState state, SparseSubsetTarget target, RandomEngine random) {
		validate(state, target); if (random == null) throw new IllegalArgumentException("random engine required");
		int inactive = target.candidateCount() - state.size(); if (inactive == 0) throw new IllegalArgumentException("full sparse model has no inactive candidate");
		int choice = random.nextInt(inactive), candidate = -1;
		for (int i = 0; i < target.candidateCount(); i++) if (!state.active(i) && choice-- == 0) { candidate = i; break; }
		return new SparseCandidateChoice(candidate, -Math.log(inactive));
	}
	@Override public double logProbability(int candidate, SparseSubsetState state, SparseSubsetTarget target) {
		validate(state, target); if (candidate < 0 || candidate >= target.candidateCount() || state.active(candidate)) return Double.NEGATIVE_INFINITY;
		return -Math.log(target.candidateCount() - state.size());
	}
	private static void validate(SparseSubsetState state, SparseSubsetTarget target) { if (state == null || target == null) throw new IllegalArgumentException("state and target required"); target.validate(state); }
}
