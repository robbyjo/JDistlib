/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Restored sparse checkpoint plus its model/options fingerprints and platform. */
public final class PortableSparseSubsetCheckpoint {
	private final SparseSubsetCheckpoint checkpoint; private final String modelFingerprint, optionsFingerprint, platform;
	PortableSparseSubsetCheckpoint(SparseSubsetCheckpoint checkpoint, String modelFingerprint, String optionsFingerprint, String platform) {
		this.checkpoint = checkpoint; this.modelFingerprint = modelFingerprint; this.optionsFingerprint = optionsFingerprint; this.platform = platform;
	}
	public SparseSubsetCheckpoint checkpoint() { return checkpoint; }
	public String modelFingerprint() { return modelFingerprint; }
	public String optionsFingerprint() { return optionsFingerprint; }
	public String platform() { return platform; }
}
