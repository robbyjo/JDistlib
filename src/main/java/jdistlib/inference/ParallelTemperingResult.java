/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Cold-chain draws and adjacent-temperature swap diagnostics. */
public final class ParallelTemperingResult {
	private final ChainResult coldChain;
	private final int[] attempted, accepted;
	ParallelTemperingResult(ChainResult coldChain, int[] attempted, int[] accepted) {
		this.coldChain = coldChain; this.attempted = attempted.clone();
		this.accepted = accepted.clone();
	}
	public ChainResult coldChain() { return coldChain; }
	public int[] attemptedSwaps() { return attempted.clone(); }
	public int[] acceptedSwaps() { return accepted.clone(); }
	public double swapAcceptance(int pair) {
		return attempted[pair] == 0 ? Double.NaN : (double) accepted[pair] / attempted[pair];
	}
}
