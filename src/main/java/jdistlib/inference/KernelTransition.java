/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable result of one reusable Markov transition. */
public final class KernelTransition<S> {
	private final S state;
	private final double[] position;
	private final double logDensity;
	private final IterationStats statistics;
	public KernelTransition(S state, double[] position, double logDensity,
			IterationStats statistics) {
		if (state == null || position == null || statistics == null)
			throw new IllegalArgumentException("transition fields are required");
		this.state = state; this.position = position.clone();
		this.logDensity = logDensity; this.statistics = statistics;
	}
	public S state() { return state; }
	public double[] position() { return position.clone(); }
	public double logDensity() { return logDensity; }
	public IterationStats statistics() { return statistics; }
}
