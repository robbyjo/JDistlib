/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Gaussian Metropolis sweeps with independently adapted coordinate scales. */
public final class ComponentWiseMetropolis implements Sampler {
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
		double[] logSteps = new double[state.length];
		Arrays.fill(logSteps, Math.log(options.stepSize()));
		double acceptanceSum = 0.0;
		int completed = 0;
		int total = options.warmupIterations()
				+ options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return output.result(state, value, completed,
					random, warmup(options, logSteps, acceptanceSum), ChainResult.Status.CANCELLED);
			double sweepProbability = 0.0;
			boolean anyAccepted = false;
			for (int coordinate = 0; coordinate < state.length; coordinate++) {
				double old = state[coordinate];
				state[coordinate] = old + Math.exp(logSteps[coordinate])
						* random.nextGaussian();
				double proposed = target.logDensity(state);
				double probability = Double.isFinite(proposed)
						? Math.min(1.0, Math.exp(proposed - value)) : 0.0;
				if (random.nextDouble() < probability) {
					value = proposed; anyAccepted = true;
				} else state[coordinate] = old;
				sweepProbability += probability;
				if (iteration < options.warmupIterations() && options.adaptStepSize()) {
					logSteps[coordinate] += (probability - options.targetAcceptance())
							/ Math.sqrt(iteration + 10.0);
				}
			}
			sweepProbability /= state.length;
			completed++;
			if (iteration < options.warmupIterations()) acceptanceSum += sweepProbability;
			else if ((iteration - options.warmupIterations() + 1) % options.thinning() == 0)
				output.add(state, value, new IterationStats(anyAccepted, sweepProbability,
						geometricMean(logSteps), -value, Double.NaN, false, 0, 0));
		}
		return output.result(state, value, completed, random,
				warmup(options, logSteps, acceptanceSum), ChainResult.Status.SUCCESS);
	}

	private static double geometricMean(double[] logValues) {
		double sum = 0.0;
		for (double value : logValues) sum += value;
		return Math.exp(sum / logValues.length);
	}
	private static WarmupResult warmup(SamplingOptions options, double[] logSteps,
			double acceptanceSum) {
		return new WarmupResult(options.warmupIterations(), options.stepSize(),
				geometricMean(logSteps), null, options.warmupIterations() == 0
				? Double.NaN : acceptanceSum / options.warmupIterations());
	}
}
