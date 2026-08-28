/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Checkpoint plus the fingerprints and platform metadata needed to validate a resume. */
public final class PortableCheckpoint {
	private final ChainCheckpoint checkpoint; private final String modelFingerprint, optionsFingerprint, platform;
	PortableCheckpoint(ChainCheckpoint checkpoint, String modelFingerprint, String optionsFingerprint, String platform) {
		this.checkpoint = checkpoint; this.modelFingerprint = modelFingerprint; this.optionsFingerprint = optionsFingerprint; this.platform = platform; }
	public ChainCheckpoint checkpoint() { return checkpoint; } public String modelFingerprint() { return modelFingerprint; }
	public String optionsFingerprint() { return optionsFingerprint; } public String platform() { return platform; }
}
