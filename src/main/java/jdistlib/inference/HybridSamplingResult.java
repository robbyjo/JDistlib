/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Retained mixed-state chain and diagnostics from its scheduled kernels. */
public final class HybridSamplingResult {
	private final ChainResult chain; private final HybridSamplerDiagnostics diagnostics;
	HybridSamplingResult(ChainResult chain, HybridSamplerDiagnostics diagnostics) { this.chain = chain; this.diagnostics = diagnostics; }
	public ChainResult chain() { return chain; }
	public HybridSamplerDiagnostics diagnostics() { return diagnostics; }
}
