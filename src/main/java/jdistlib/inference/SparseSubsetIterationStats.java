/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Per-transition statistics for sparse subset RJMCMC. */
public final class SparseSubsetIterationStats {
	private final int fromSize, toSize; private final String move;
	private final boolean jumpAccepted, invalidProposal, withinAccepted;
	private final double jumpAcceptanceProbability, logAcceptanceRatio;
	public SparseSubsetIterationStats(int fromSize, int toSize, String move, boolean jumpAccepted,
			boolean invalidProposal, boolean withinAccepted, double jumpAcceptanceProbability,
			double logAcceptanceRatio) {
		if (fromSize < 0 || toSize < 0 || move == null || jumpAcceptanceProbability < 0.0
				|| jumpAcceptanceProbability > 1.0 || !Double.isFinite(jumpAcceptanceProbability)
				|| Double.isNaN(logAcceptanceRatio)) throw new IllegalArgumentException("valid sparse transition statistics required");
		this.fromSize = fromSize; this.toSize = toSize; this.move = move; this.jumpAccepted = jumpAccepted;
		this.invalidProposal = invalidProposal; this.withinAccepted = withinAccepted;
		this.jumpAcceptanceProbability = jumpAcceptanceProbability; this.logAcceptanceRatio = logAcceptanceRatio;
	}
	public int fromSize() { return fromSize; }
	public int toSize() { return toSize; }
	public String move() { return move; }
	public boolean jumpAccepted() { return jumpAccepted; }
	public boolean invalidProposal() { return invalidProposal; }
	public boolean withinAccepted() { return withinAccepted; }
	public double jumpAcceptanceProbability() { return jumpAcceptanceProbability; }
	public double logAcceptanceRatio() { return logAcceptanceRatio; }
}
