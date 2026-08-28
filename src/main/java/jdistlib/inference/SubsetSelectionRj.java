/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Factory for the standard add/drop/swap subset-selection RJ schedule. */
public final class SubsetSelectionRj {
	private SubsetSelectionRj() {}
	public static ReversibleJumpSampler sampler(RjBirthProposal birthProposal,
			double withinModelScale, double withinModelTargetAcceptance) {
		if (birthProposal == null) throw new IllegalArgumentException("birth proposal required");
		ReversibleJumpMove[] moves = {new SubsetBirthMove(birthProposal), new SubsetDeathMove(birthProposal), new SubsetSwapMove(birthProposal)};
		return new ReversibleJumpSampler(moves, new double[] {1.0, 1.0, 0.5},
				new AdaptiveRjRandomWalkKernel("within-model-random-walk", withinModelScale, withinModelTargetAcceptance));
	}
}
