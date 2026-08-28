/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable ragged RJMCMC draws, transition statistics, diagnostics inputs, and restart state. */
public final class ReversibleJumpResult {
	public enum Status { SUCCESS, CANCELLED, INVALID_INITIAL_STATE, NUMERICAL_FAILURE }
	private final ReversibleJumpState[] draws; private final double[] logJoints; private final ReversibleJumpIterationStats[] statistics;
	private final ReversibleJumpCheckpoint checkpoint; private final Status status; private final List<String> warnings;
	private final String[] moveNames; private final long[] moveAttempts, moveAccepts, invalidProposals;
	ReversibleJumpResult(ReversibleJumpState[] draws, double[] logJoints, ReversibleJumpIterationStats[] statistics,
			ReversibleJumpCheckpoint checkpoint, Status status, List<String> warnings, String[] moveNames,
			long[] moveAttempts, long[] moveAccepts, long[] invalidProposals) {
		if (draws == null || logJoints == null || statistics == null || draws.length != logJoints.length
				|| draws.length != statistics.length || checkpoint == null || status == null || warnings == null
				|| moveNames == null || moveAttempts == null || moveAccepts == null || invalidProposals == null
				|| moveNames.length != moveAttempts.length || moveNames.length != moveAccepts.length || moveNames.length != invalidProposals.length)
			throw new IllegalArgumentException("matching RJ result fields required");
		this.draws = draws.clone(); this.logJoints = logJoints.clone(); this.statistics = statistics.clone();
		this.checkpoint = checkpoint; this.status = status; this.warnings = Collections.unmodifiableList(new ArrayList<String>(warnings));
		this.moveNames = moveNames.clone(); this.moveAttempts = moveAttempts.clone(); this.moveAccepts = moveAccepts.clone(); this.invalidProposals = invalidProposals.clone();
	}
	public int size() { return draws.length; }
	public ReversibleJumpState draw(int index) { return draws[index]; }
	public ReversibleJumpState[] draws() { return draws.clone(); }
	/** Returns rectangular parameters for draws in one model, preserving retained order. */
	public double[][] drawsForModel(long modelId) {
		int count = 0; for (ReversibleJumpState draw : draws) if (draw.modelId() == modelId) count++;
		double[][] result = new double[count][]; int offset = 0;
		for (ReversibleJumpState draw : draws) if (draw.modelId() == modelId) result[offset++] = draw.parameters();
		return result;
	}
	public double logJointAt(int index) { return logJoints[index]; }
	public double[] logJoints() { return logJoints.clone(); }
	public ReversibleJumpIterationStats statisticsAt(int index) { return statistics[index]; }
	public ReversibleJumpIterationStats[] statistics() { return statistics.clone(); }
	public ReversibleJumpCheckpoint checkpoint() { return checkpoint; }
	public Status status() { return status; }
	public List<String> warnings() { return warnings; }
	public int moveCount() { return moveNames.length; }
	public String moveName(int index) { return moveNames[index]; }
	public long moveAttempts(int index) { return moveAttempts[index]; }
	public long moveAccepts(int index) { return moveAccepts[index]; }
	public long invalidProposals(int index) { return invalidProposals[index]; }
	public double moveAcceptanceRate(int index) { return moveAttempts[index] == 0L ? Double.NaN : moveAccepts[index] / (double) moveAttempts[index]; }
}
