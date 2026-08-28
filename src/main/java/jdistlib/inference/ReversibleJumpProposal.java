/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Transactional RJ proposal including all non-schedule Hastings terms. */
public final class ReversibleJumpProposal {
	private final ReversibleJumpState proposedState; private final String reverseMove, rejectionReason;
	private final double logForwardDensity, logReverseDensity, logAbsJacobian;
	private ReversibleJumpProposal(ReversibleJumpState proposedState, String reverseMove,
			double logForwardDensity, double logReverseDensity, double logAbsJacobian, String rejectionReason) {
		this.proposedState = proposedState; this.reverseMove = reverseMove;
		this.logForwardDensity = logForwardDensity; this.logReverseDensity = logReverseDensity;
		this.logAbsJacobian = logAbsJacobian; this.rejectionReason = rejectionReason;
	}
	public static ReversibleJumpProposal valid(ReversibleJumpState proposedState, String reverseMove,
			double logForwardDensity, double logReverseDensity, double logAbsJacobian) {
		if (proposedState == null || reverseMove == null || reverseMove.trim().isEmpty()
				|| !Double.isFinite(logForwardDensity) || !Double.isFinite(logReverseDensity)
				|| !Double.isFinite(logAbsJacobian)) throw new IllegalArgumentException("complete finite RJ proposal terms required");
		return new ReversibleJumpProposal(proposedState, reverseMove, logForwardDensity,
				logReverseDensity, logAbsJacobian, null);
	}
	public static ReversibleJumpProposal invalid(String reason) {
		if (reason == null || reason.trim().isEmpty()) throw new IllegalArgumentException("rejection reason required");
		return new ReversibleJumpProposal(null, null, Double.NaN, Double.NaN, Double.NaN, reason);
	}
	public boolean valid() { return rejectionReason == null; }
	public ReversibleJumpState proposedState() { return proposedState; }
	public String reverseMove() { return reverseMove; }
	public double logForwardDensity() { return logForwardDensity; }
	public double logReverseDensity() { return logReverseDensity; }
	public double logAbsJacobian() { return logAbsJacobian; }
	public String rejectionReason() { return rejectionReason; }
}
