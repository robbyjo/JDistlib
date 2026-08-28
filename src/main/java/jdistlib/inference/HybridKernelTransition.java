/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Outcome of one hybrid-kernel update. */
public final class HybridKernelTransition {
	private final double logDensity, acceptanceProbability; private final boolean accepted, supportRejected;
	public HybridKernelTransition(double logDensity, boolean accepted, double acceptanceProbability, boolean supportRejected) {
		if (Double.isNaN(logDensity) || acceptanceProbability < 0.0 || acceptanceProbability > 1.0)
			throw new IllegalArgumentException("invalid transition outcome");
		this.logDensity = logDensity; this.accepted = accepted; this.acceptanceProbability = acceptanceProbability; this.supportRejected = supportRejected;
	}
	public double logDensity() { return logDensity; }
	public boolean accepted() { return accepted; }
	public double acceptanceProbability() { return acceptanceProbability; }
	public boolean supportRejected() { return supportRejected; }
}
