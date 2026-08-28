/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Posterior summary conditional on a ragged parameter being present. */
public final class ReversibleJumpParameterSummary {
	private final String name; private final long draws; private final double mean, standardDeviation;
	ReversibleJumpParameterSummary(String name, long draws, double mean, double standardDeviation) {
		this.name = name; this.draws = draws; this.mean = mean; this.standardDeviation = standardDeviation;
	}
	public String name() { return name; }
	public long draws() { return draws; }
	public double mean() { return mean; }
	public double standardDeviation() { return standardDeviation; }
}
