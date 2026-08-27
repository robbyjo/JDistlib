/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Composes exact, adaptive-rejection, Metropolis, or blocked Gibbs kernels. */
public final class GibbsSampler implements Sampler {
	private final GibbsKernel[] kernels;
	public GibbsSampler(GibbsKernel... kernels) {
		if (kernels == null || kernels.length == 0)
			throw new IllegalArgumentException("at least one Gibbs kernel is required");
		this.kernels = kernels.clone();
		for (GibbsKernel kernel : this.kernels)
			if (kernel == null) throw new IllegalArgumentException("kernels must not contain null");
	}

	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || initialState.length == 0
				|| options == null || random == null)
			throw new IllegalArgumentException("target, state, options and random are required");
		target = SamplerTargets.local(target);
		double[] state = initialState.clone();
		double value = target.logDensity(state);
		ChainAccumulator output = new ChainAccumulator();
		if (!Double.isFinite(value)) {
			output.warn("initial log density is not finite");
			return output.result(state, value, 0, random, null,
					ChainResult.Status.INVALID_INITIAL_STATE);
		}
		int completed = 0;
		int total = options.warmupIterations()
				+ options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return output.result(state, value, completed,
					random, warmup(options), ChainResult.Status.CANCELLED);
			for (GibbsKernel kernel : kernels) kernel.update(state, target, random);
			value = target.logDensity(state);
			if (!Double.isFinite(value)) {
				output.warn("Gibbs kernel produced a non-finite state at iteration " + iteration);
				return output.result(state, value, completed, random, warmup(options),
						ChainResult.Status.NUMERICAL_FAILURE);
			}
			completed++;
			if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0)
				output.add(state, value, new IterationStats(true, 1.0, Double.NaN,
						-value, Double.NaN, false, 0, 0));
		}
		return output.result(state, value, completed, random, warmup(options),
				ChainResult.Status.SUCCESS);
	}
	private static WarmupResult warmup(SamplingOptions options) {
		return new WarmupResult(options.warmupIterations(), Double.NaN,
				Double.NaN, null, 1.0);
	}
}
