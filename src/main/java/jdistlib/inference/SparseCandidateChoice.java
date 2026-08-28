/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Candidate selected by a normalized sparse birth proposal. */
public final class SparseCandidateChoice {
	private final int candidate; private final double logProbability;
	public SparseCandidateChoice(int candidate, double logProbability) {
		if (candidate < 0 || !Double.isFinite(logProbability) || logProbability > 0.0) throw new IllegalArgumentException("valid candidate probability required");
		this.candidate = candidate; this.logProbability = logProbability;
	}
	public int candidate() { return candidate; }
	public double logProbability() { return logProbability; }
}
