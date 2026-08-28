/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** In-memory state-and-stream restart point including a cloned random engine. */
public final class ChainCheckpoint {
	private final double[] state;
	private final double logDensity;
	private final int completedIterations;
	private final RandomEngine random;
	private final SamplerCheckpoint samplerCheckpoint;

	public ChainCheckpoint(double[] state, double logDensity,
			int completedIterations, RandomEngine random) {
		this(state, logDensity, completedIterations, random, null);
	}

	public ChainCheckpoint(double[] state, double logDensity,
			int completedIterations, RandomEngine random,
			SamplerCheckpoint samplerCheckpoint) {
		if (state == null || random == null || completedIterations < 0)
			throw new IllegalArgumentException("state, iteration and random stream are required");
		this.state = state.clone();
		this.logDensity = logDensity;
		this.completedIterations = completedIterations;
		this.random = random.clone();
		this.samplerCheckpoint = samplerCheckpoint;
	}
	public double[] state() { return state.clone(); }
	public double logDensity() { return logDensity; }
	public int completedIterations() { return completedIterations; }
	public RandomEngine random() { return random.clone(); }
	public SamplerCheckpoint samplerCheckpoint() { return samplerCheckpoint; }
}
