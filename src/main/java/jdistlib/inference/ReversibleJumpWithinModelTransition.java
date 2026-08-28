/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Outcome of one within-model transition in an RJ schedule. */
public final class ReversibleJumpWithinModelTransition {
	private final ReversibleJumpState state; private final double logJoint, acceptanceProbability; private final boolean accepted;
	public ReversibleJumpWithinModelTransition(ReversibleJumpState state, double logJoint,
			boolean accepted, double acceptanceProbability) {
		if (state == null || Double.isNaN(logJoint) || acceptanceProbability < 0.0 || acceptanceProbability > 1.0
				|| !Double.isFinite(acceptanceProbability))
			throw new IllegalArgumentException("valid within-model transition required");
		this.state = state; this.logJoint = logJoint; this.accepted = accepted; this.acceptanceProbability = acceptanceProbability;
	}
	public ReversibleJumpState state() { return state; }
	public double logJoint() { return logJoint; }
	public boolean accepted() { return accepted; }
	public double acceptanceProbability() { return acceptanceProbability; }
}
