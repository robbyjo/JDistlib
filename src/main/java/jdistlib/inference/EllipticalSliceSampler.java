/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Tuning-free elliptical slice sampler; target is the likelihood-only log density. */
public final class EllipticalSliceSampler implements Sampler {
	private final GaussianReference reference;
	public EllipticalSliceSampler(GaussianReference reference) {
		if (reference == null) throw new IllegalArgumentException("Gaussian reference required");
		this.reference = reference;
	}
	@Override public ChainResult sample(LogDensity likelihood, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		double[] mean = reference.mean(); double[] state = initialState.clone();
		if (mean.length != state.length) throw new IllegalArgumentException("reference dimension mismatch");
		double value = likelihood.logDensity(state); ChainAccumulator output = new ChainAccumulator();
		int total = options.warmupIterations() + options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return output.result(state, value, iteration, random,
					new WarmupResult(options.warmupIterations(), Double.NaN, Double.NaN,
							null, 1.0), ChainResult.Status.CANCELLED);
			double[] nu = reference.random(random);
			for (int i = 0; i < nu.length; i++) nu[i] -= mean[i];
			double threshold = value + Math.log(Math.max(Double.MIN_VALUE, random.nextDouble()));
			double angle = 2.0 * Math.PI * random.nextDouble();
			double lower = angle - 2.0 * Math.PI, upper = angle;
			double[] proposal = new double[state.length]; double proposed = Double.NEGATIVE_INFINITY;
			int attempts = 0;
			while (attempts++ < options.maximumSliceSteps()) {
				for (int i = 0; i < state.length; i++) proposal[i] = mean[i]
						+ (state[i] - mean[i]) * Math.cos(angle) + nu[i] * Math.sin(angle);
				proposed = likelihood.logDensity(proposal);
				if (proposed >= threshold) break;
				if (angle < 0.0) lower = angle; else upper = angle;
				angle = lower + random.nextDouble() * (upper - lower);
			}
			boolean moved = proposed >= threshold;
			if (moved) { state = proposal.clone(); value = proposed; }
			IterationStats stats = new IterationStats(moved, 1.0,
					Double.NaN, Double.NaN, Double.NaN, false, 0, attempts);
			if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0)
				output.retain(options, state, value, stats);
			options.progress(iteration + 1, total,
					iteration < options.warmupIterations(), stats);
		}
		return output.result(state, value, total, random,
				new WarmupResult(options.warmupIterations(), Double.NaN, Double.NaN,
						null, 1.0), ChainResult.Status.SUCCESS);
	}
}
