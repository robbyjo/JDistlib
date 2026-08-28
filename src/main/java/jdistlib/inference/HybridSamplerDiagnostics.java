/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Per-kernel acceptance and support diagnostics for a hybrid schedule. */
public final class HybridSamplerDiagnostics {
	private final String[] names; private final long[] attempts, accepts, supportRejections;
	HybridSamplerDiagnostics(String[] names, long[] attempts, long[] accepts, long[] supportRejections) {
		this.names = names; this.attempts = attempts; this.accepts = accepts; this.supportRejections = supportRejections;
	}
	public int kernelCount() { return names.length; }
	public String kernelName(int kernel) { return names[kernel]; }
	public long attempts(int kernel) { return attempts[kernel]; }
	public long accepts(int kernel) { return accepts[kernel]; }
	public long supportRejections(int kernel) { return supportRejections[kernel]; }
	public double acceptanceRate(int kernel) { return attempts[kernel] == 0L ? Double.NaN : accepts[kernel] / (double) attempts[kernel]; }
}
