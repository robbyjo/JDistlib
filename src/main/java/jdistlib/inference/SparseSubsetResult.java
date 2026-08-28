/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** One bounded sparse RJ segment plus its exact continuation checkpoint. */
public final class SparseSubsetResult {
	public enum Status { SUCCESS, CANCELLED, INVALID_INITIAL_STATE, NUMERICAL_FAILURE }
	private final SparseSubsetState[] draws; private final double[] logJoints; private final SparseSubsetIterationStats[] statistics;
	private final SparseSubsetCheckpoint checkpoint; private final Status status; private final List<String> warnings;
	SparseSubsetResult(SparseSubsetState[] draws, double[] logJoints, SparseSubsetIterationStats[] statistics,
			SparseSubsetCheckpoint checkpoint, Status status, List<String> warnings) {
		if (draws == null || logJoints == null || statistics == null || draws.length != logJoints.length
				|| draws.length != statistics.length || checkpoint == null || status == null || warnings == null)
			throw new IllegalArgumentException("matching sparse result fields required");
		this.draws = draws.clone(); this.logJoints = logJoints.clone(); this.statistics = statistics.clone();
		this.checkpoint = checkpoint; this.status = status; this.warnings = Collections.unmodifiableList(new ArrayList<String>(warnings));
	}
	public int size() { return draws.length; }
	public SparseSubsetState draw(int index) { return draws[index]; }
	public SparseSubsetState[] draws() { return draws.clone(); }
	public double logJointAt(int index) { return logJoints[index]; }
	public double[] logJoints() { return logJoints.clone(); }
	public SparseSubsetIterationStats statisticsAt(int index) { return statistics[index]; }
	public SparseSubsetIterationStats[] statistics() { return statistics.clone(); }
	public SparseSubsetCheckpoint checkpoint() { return checkpoint; }
	public Status status() { return status; }
	public List<String> warnings() { return warnings; }
}
