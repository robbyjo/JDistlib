/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Coordinate-wise stepping-out and shrinkage slice sampler. */
public final class SliceSampler implements Sampler {
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
			for (int coordinate = 0; coordinate < state.length; coordinate++)
				value = update(target, state, value, coordinate, options, random);
			completed++;
			if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0)
				output.add(state, value, new IterationStats(true, 1.0,
						options.sliceWidth(), -value, Double.NaN, false, 0, 0));
		}
		return output.result(state, value, completed, random, warmup(options),
				ChainResult.Status.SUCCESS);
	}

	private static double update(LogDensity target, double[] state, double value,
			int coordinate, SamplingOptions options, RandomEngine random) {
		double original = state[coordinate];
		double logSlice = value + Math.log(random.nextDouble());
		double left = original - options.sliceWidth() * random.nextDouble();
		double right = left + options.sliceWidth();
		int leftBudget = random.nextInt(options.maximumSliceSteps());
		int rightBudget = options.maximumSliceSteps() - 1 - leftBudget;
		state[coordinate] = left;
		while (leftBudget-- > 0 && target.logDensity(state) > logSlice) {
			left -= options.sliceWidth(); state[coordinate] = left;
		}
		state[coordinate] = right;
		while (rightBudget-- > 0 && target.logDensity(state) > logSlice) {
			right += options.sliceWidth(); state[coordinate] = right;
		}
		for (int attempt = 0; attempt < options.maximumSliceSteps() * 10; attempt++) {
			double candidate = left + random.nextDouble() * (right - left);
			state[coordinate] = candidate;
			double proposed = target.logDensity(state);
			if (Double.isFinite(proposed) && proposed >= logSlice) return proposed;
			if (candidate < original) left = candidate; else right = candidate;
		}
		state[coordinate] = original;
		return value;
	}

	private static WarmupResult warmup(SamplingOptions options) {
		return new WarmupResult(options.warmupIterations(), options.sliceWidth(),
				options.sliceWidth(), null, 1.0);
	}
}
