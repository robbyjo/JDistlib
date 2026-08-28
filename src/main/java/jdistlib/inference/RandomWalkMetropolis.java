/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Isotropic Gaussian random-walk Metropolis with warmup scale adaptation. */
public final class RandomWalkMetropolis implements Sampler {
	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || initialState.length == 0
				|| options == null || random == null)
			throw new IllegalArgumentException("target, state, options and random are required");
		target = SamplerTargets.local(target);
		double[] state = initialState.clone();
		double[] proposal = new double[state.length];
		double value = target.logDensity(state);
		ChainAccumulator output = new ChainAccumulator();
		if (!Double.isFinite(value)) {
			output.warn("initial log density is not finite");
			return output.result(state, value, 0, random, null,
					ChainResult.Status.INVALID_INITIAL_STATE);
		}
		double initialStep = options.stepSize();
		double logStep = Math.log(initialStep);
		double acceptanceSum = 0.0;
		int completed = 0;
		int total = options.warmupIterations()
				+ options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled())
				return output.result(state, value, completed, random,
						warmup(options, initialStep, Math.exp(logStep), acceptanceSum),
						ChainResult.Status.CANCELLED);
			System.arraycopy(state, 0, proposal, 0, state.length);
			double step = Math.exp(logStep);
			for (int i = 0; i < proposal.length; i++)
				proposal[i] += step * random.nextGaussian();
			double proposed = target.logDensity(proposal);
			double probability = Double.isFinite(proposed)
					? Math.min(1.0, Math.exp(proposed - value)) : 0.0;
			boolean accepted = random.nextDouble() < probability;
			if (accepted) {
				double[] previous = state;
				state = proposal;
				proposal = previous;
				value = proposed;
			}
			completed++;
			IterationStats stats = new IterationStats(accepted, probability,
					step, -value, Double.NaN, false, 0, 0);
			if (iteration < options.warmupIterations()) {
				acceptanceSum += probability;
				if (options.adaptStepSize()) {
					double rate = 1.0 / Math.sqrt(iteration + 10.0);
					logStep += rate * (probability - options.targetAcceptance());
				}
			} else if ((iteration - options.warmupIterations() + 1)
					% options.thinning() == 0) {
				output.retain(options, state, value, stats);
			}
			options.progress(completed, total, iteration < options.warmupIterations(), stats);
		}
		return output.result(state, value, completed, random,
				warmup(options, initialStep, Math.exp(logStep), acceptanceSum),
				ChainResult.Status.SUCCESS);
	}

	private static WarmupResult warmup(SamplingOptions options, double initial,
			double current, double acceptanceSum) {
		double mean = options.warmupIterations() == 0 ? Double.NaN
				: acceptanceSum / options.warmupIterations();
		return new WarmupResult(options.warmupIterations(), initial, current, null, mean);
	}
}
