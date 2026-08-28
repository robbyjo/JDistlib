/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;

/** An approximate composed law together with strategy, error, and seed provenance. */
public final class DistributionApproximation {
	private final GenericDistribution distribution;
	private final NumericalEstimate diagnostics;
	private final long seed;

	public DistributionApproximation(GenericDistribution distribution,
			NumericalEstimate diagnostics, long seed) {
		if (distribution == null || diagnostics == null)
			throw new IllegalArgumentException("distribution and diagnostics are required");
		this.distribution = distribution;
		this.diagnostics = diagnostics;
		this.seed = seed;
	}
	public GenericDistribution getDistribution() { return distribution; }
	public NumericalEstimate getDiagnostics() { return diagnostics; }
	public long getSeed() { return seed; }
}
