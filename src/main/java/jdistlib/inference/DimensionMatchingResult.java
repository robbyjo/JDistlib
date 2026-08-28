/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** One side of a dimension-matching map: state plus complementary auxiliaries. */
public final class DimensionMatchingResult {
	private final ReversibleJumpState state; private final double[] auxiliary;
	public DimensionMatchingResult(ReversibleJumpState state, double... auxiliary) {
		if (state == null || auxiliary == null) throw new IllegalArgumentException("state and auxiliary values required");
		this.state = state; this.auxiliary = auxiliary.clone();
		for (double value : this.auxiliary) if (!Double.isFinite(value)) throw new IllegalArgumentException("auxiliary values must be finite");
	}
	public ReversibleJumpState state() { return state; }
	public double[] auxiliary() { return auxiliary.clone(); }
}
