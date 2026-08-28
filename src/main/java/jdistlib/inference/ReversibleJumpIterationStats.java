/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Per-iteration within-model and trans-dimensional transition statistics. */
public final class ReversibleJumpIterationStats {
	private final long fromModel, toModel; private final String move;
	private final boolean jumpAttempted, jumpAccepted, invalidProposal;
	private final double jumpAcceptanceProbability, logAcceptanceRatio;
	private final int withinAttempts, withinAccepts;
	public ReversibleJumpIterationStats(long fromModel, long toModel, String move,
			boolean jumpAttempted, boolean jumpAccepted, boolean invalidProposal,
			double jumpAcceptanceProbability, double logAcceptanceRatio,
			int withinAttempts, int withinAccepts) {
		if (fromModel < 0L || toModel < 0L || withinAttempts < 0 || withinAccepts < 0 || withinAccepts > withinAttempts
				|| jumpAcceptanceProbability < 0.0 || jumpAcceptanceProbability > 1.0
				|| !Double.isFinite(jumpAcceptanceProbability) || Double.isNaN(logAcceptanceRatio))
			throw new IllegalArgumentException("invalid RJ iteration statistics");
		this.fromModel = fromModel; this.toModel = toModel; this.move = move; this.jumpAttempted = jumpAttempted;
		this.jumpAccepted = jumpAccepted; this.invalidProposal = invalidProposal;
		this.jumpAcceptanceProbability = jumpAcceptanceProbability; this.logAcceptanceRatio = logAcceptanceRatio;
		this.withinAttempts = withinAttempts; this.withinAccepts = withinAccepts;
	}
	public long fromModel() { return fromModel; }
	public long toModel() { return toModel; }
	public String move() { return move; }
	public boolean jumpAttempted() { return jumpAttempted; }
	public boolean jumpAccepted() { return jumpAccepted; }
	public boolean invalidProposal() { return invalidProposal; }
	public double jumpAcceptanceProbability() { return jumpAcceptanceProbability; }
	public double logAcceptanceRatio() { return logAcceptanceRatio; }
	public int withinAttempts() { return withinAttempts; }
	public int withinAccepts() { return withinAccepts; }
}
