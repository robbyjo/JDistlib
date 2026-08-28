/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Adapts one frozen ordinary JDistlib sampler transition for use within an RJ model. */
public final class FixedDimensionSamplerRjKernel implements ReversibleJumpWithinModelKernel {
	private final String name; private final Sampler sampler; private final SamplingOptions options;
	public FixedDimensionSamplerRjKernel(String name, Sampler sampler, SamplingOptions transitionOptions) {
		if (name == null || name.trim().isEmpty() || sampler == null || transitionOptions == null
				|| transitionOptions.warmupIterations() != 0 || transitionOptions.sampleIterations() != 1
				|| transitionOptions.thinning() != 1 || !transitionOptions.storeDraws())
			throw new IllegalArgumentException("a named sampler configured for one stored, unthinned, zero-warmup transition is required");
		this.name = name; this.sampler = sampler; this.options = transitionOptions;
	}
	@Override public String name() { return name; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) { return state != null && state.dimension() > 0; }
	@Override public ReversibleJumpWithinModelTransition update(ReversibleJumpState state, double currentLogJoint,
			ReversibleJumpTarget target, RandomEngine random, boolean warmup) {
		ChainResult chain = sampler.sample(target.fixedModelTarget(state.modelId()), state.parameters(), options, random);
		if (chain.status() != ChainResult.Status.SUCCESS || chain.size() != 1)
			throw new IllegalStateException("fixed-dimensional RJ kernel failed: " + chain.status());
		IterationStats stats = chain.statisticsAt(0); ReversibleJumpState result = new ReversibleJumpState(state.modelId(), chain.sample(0));
		return new ReversibleJumpWithinModelTransition(result, chain.logDensityAt(0), stats.accepted(), stats.acceptanceProbability());
	}
}
