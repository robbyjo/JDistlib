/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** First-class inference result combining chains, diagnostics, and provenance. */
public final class Fit {
	private final ChainResult[] chains;
	private final McmcDiagnosticReport diagnostics;
	private final RunManifest manifest;
	Fit(ChainResult[] chains, McmcDiagnosticReport diagnostics, RunManifest manifest) {
		this.chains = chains.clone(); this.diagnostics = diagnostics; this.manifest = manifest;
	}
	public ChainResult[] chains() { return chains.clone(); }
	public McmcDiagnosticReport diagnostics() { return diagnostics; }
	public RunManifest manifest() { return manifest; }
	public boolean healthy() { return diagnostics.sampler().healthy() && diagnostics.warnings().isEmpty(); }
}
