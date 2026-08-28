/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable factor timing and numerical-stability snapshot. */
public final class FactorProfile {
	private final String name; private final long calls, elapsedNanoseconds, nonFiniteResults, estimatedAllocatedBytes;
	FactorProfile(String name, long calls, long elapsedNanoseconds, long nonFiniteResults, long estimatedAllocatedBytes) { this.name = name; this.calls = calls; this.elapsedNanoseconds = elapsedNanoseconds; this.nonFiniteResults = nonFiniteResults; this.estimatedAllocatedBytes = estimatedAllocatedBytes; }
	public String name() { return name; } public long calls() { return calls; } public long elapsedNanoseconds() { return elapsedNanoseconds; }
	public long nonFiniteResults() { return nonFiniteResults; } public double meanNanoseconds() { return calls == 0 ? Double.NaN : (double) elapsedNanoseconds / calls; }
	/** Best-effort positive heap-growth estimate; use a profiler for exact allocation attribution. */
	public long estimatedAllocatedBytes() { return estimatedAllocatedBytes; }
}
