/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.AdaptiveRejectionSampler;
import jdistlib.rng.RandomEngine;

/** Rebuilds and draws a caller-certified log-concave full conditional. */
public final class AdaptiveRejectionGibbsKernel implements GibbsKernel {
	@FunctionalInterface
	public interface ConditionalFactory {
		AdaptiveRejectionSampler create(double[] currentState);
	}
	private final int coordinate;
	private final ConditionalFactory factory;
	public AdaptiveRejectionGibbsKernel(int coordinate, ConditionalFactory factory) {
		if (coordinate < 0 || factory == null)
			throw new IllegalArgumentException("coordinate and conditional factory are required");
		this.coordinate = coordinate; this.factory = factory;
	}
	@Override public void update(double[] state, LogDensity target, RandomEngine random) {
		if (coordinate >= state.length) throw new IllegalArgumentException("coordinate is outside state");
		AdaptiveRejectionSampler sampler = factory.create(state.clone());
		if (sampler == null) throw new IllegalStateException("conditional factory returned null");
		state[coordinate] = sampler.sample(random);
	}
}
