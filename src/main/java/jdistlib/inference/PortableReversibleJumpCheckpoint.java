/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Restored RJ checkpoint plus the fingerprints and platform recorded with it. */
public final class PortableReversibleJumpCheckpoint {
	private final ReversibleJumpCheckpoint checkpoint; private final String modelFingerprint, optionsFingerprint, platform;
	PortableReversibleJumpCheckpoint(ReversibleJumpCheckpoint checkpoint, String modelFingerprint, String optionsFingerprint, String platform) {
		this.checkpoint = checkpoint; this.modelFingerprint = modelFingerprint; this.optionsFingerprint = optionsFingerprint; this.platform = platform;
	}
	public ReversibleJumpCheckpoint checkpoint() { return checkpoint; }
	public String modelFingerprint() { return modelFingerprint; }
	public String optionsFingerprint() { return optionsFingerprint; }
	public String platform() { return platform; }
}
