/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Deterministically extended chain and the reason continuation stopped. */
public final class PrecisionContinuationResult {
	private final ChainResult chain; private final int chunks; private final double mcse; private final boolean goalMet;
	PrecisionContinuationResult(ChainResult chain, int chunks, double mcse, boolean goalMet) { this.chain = chain; this.chunks = chunks; this.mcse = mcse; this.goalMet = goalMet; }
	public ChainResult chain() { return chain; } public int chunks() { return chunks; } public double mcse() { return mcse; } public boolean goalMet() { return goalMet; }
}
