/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Metropolis-adjusted Langevin sampler with dual-averaged proposal scale. */
public final class MetropolisAdjustedLangevin implements Sampler {
	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		DifferentiableLogDensity density = HamiltonianSupport.gradientTarget(target, options);
		double[] state = initialState.clone(); double[] gradient = new double[state.length];
		double value = density.logDensityAndGradient(state, gradient);
		ChainAccumulator output = new ChainAccumulator();
		if (!Double.isFinite(value)) return output.result(state, value, 0, random, null,
				ChainResult.Status.INVALID_INITIAL_STATE);
		double scale = options.stepSize(); double initial = scale;
		double logScale = Math.log(scale); double acceptanceSum = 0.0;
		int total = options.warmupIterations() + options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return finish(output, state, value, iteration, random,
					options, initial, scale, acceptanceSum, ChainResult.Status.CANCELLED);
			double[] proposal = new double[state.length]; double variance = scale * scale;
			for (int i = 0; i < state.length; i++) proposal[i] = state[i]
					+ 0.5 * variance * gradient[i] + scale * random.nextGaussian();
			double[] proposedGradient = new double[state.length];
			double proposedValue = density.logDensityAndGradient(proposal, proposedGradient);
			double logRatio = proposedValue - value
					+ gaussianLogProposal(state, proposal, proposedGradient, variance)
					- gaussianLogProposal(proposal, state, gradient, variance);
			double probability = Double.isFinite(logRatio) ? Math.min(1.0, Math.exp(logRatio)) : 0.0;
			boolean accepted = random.nextDouble() < probability;
			if (accepted) { state = proposal; gradient = proposedGradient; value = proposedValue; }
			IterationStats stats = new IterationStats(accepted, probability, scale,
					Double.NaN, Double.NaN, false, 0, 1);
			if (iteration < options.warmupIterations()) {
				acceptanceSum += probability;
				if (options.adaptStepSize()) {
					double gain = 1.0 / Math.sqrt(iteration + 10.0);
					logScale += gain * (probability - 0.574); scale = Math.exp(logScale);
				}
			} else if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0) {
				output.retain(options, state, value, stats);
			}
			options.progress(iteration + 1, total,
					iteration < options.warmupIterations(), stats);
		}
		return finish(output, state, value, total, random, options, initial, scale,
				acceptanceSum, ChainResult.Status.SUCCESS);
	}
	private static double gaussianLogProposal(double to[], double from[], double[] fromGradient,
			double variance) {
		double sum = 0.0;
		for (int i = 0; i < to.length; i++) {
			double delta = to[i] - from[i] - 0.5 * variance * fromGradient[i]; sum += delta * delta;
		}
		return -0.5 * sum / variance;
	}
	private static ChainResult finish(ChainAccumulator output, double[] state, double value,
			int completed, RandomEngine random, SamplingOptions options, double initial,
			double scale, double acceptance, ChainResult.Status status) {
		WarmupResult warmup = new WarmupResult(options.warmupIterations(), initial, scale,
				null, options.warmupIterations() == 0 ? Double.NaN
				: acceptance / options.warmupIterations());
		return output.result(state, value, completed, random, warmup, status);
	}
}
