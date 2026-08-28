/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Exact gradient-informed Barker proposal with symmetric Gaussian magnitudes. */
public final class BarkerGradientSampler implements Sampler {
	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		DifferentiableLogDensity density = HamiltonianSupport.gradientTarget(target, options);
		double[] state = initialState.clone(), gradient = new double[state.length];
		double value = density.logDensityAndGradient(state, gradient);
		ChainAccumulator output = new ChainAccumulator(); double scale = options.stepSize();
		if (!Double.isFinite(value)) return output.result(state, value, 0, random, null,
				ChainResult.Status.INVALID_INITIAL_STATE);
		int total = options.warmupIterations() + options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return output.result(state, value, iteration, random,
					new WarmupResult(options.warmupIterations(), scale, scale, null, Double.NaN),
					ChainResult.Status.CANCELLED);
			double[] proposal = state.clone(); double[] delta = new double[state.length];
			for (int i = 0; i < state.length; i++) {
				double magnitude = scale * Math.abs(random.nextGaussian());
				double plus = sigmoid(gradient[i] * magnitude);
				delta[i] = random.nextDouble() < plus ? magnitude : -magnitude;
				proposal[i] += delta[i];
			}
			double[] proposedGradient = new double[state.length];
			double proposedValue = density.logDensityAndGradient(proposal, proposedGradient);
			double logRatio = proposedValue - value;
			for (int i = 0; i < state.length; i++)
				logRatio += logSigmoid(-proposedGradient[i] * delta[i])
						- logSigmoid(gradient[i] * delta[i]);
			double probability = Double.isFinite(logRatio) ? Math.min(1.0, Math.exp(logRatio)) : 0.0;
			boolean accepted = random.nextDouble() < probability;
			if (accepted) { state = proposal; gradient = proposedGradient; value = proposedValue; }
			IterationStats stats = new IterationStats(accepted, probability, scale,
					Double.NaN, Double.NaN, false, 0, 1);
			if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0)
				output.retain(options, state, value, stats);
			options.progress(iteration + 1, total,
					iteration < options.warmupIterations(), stats);
		}
		return output.result(state, value, total, random,
				new WarmupResult(options.warmupIterations(), scale, scale, null, Double.NaN),
				ChainResult.Status.SUCCESS);
	}
	private static double sigmoid(double x) {
		return x >= 0.0 ? 1.0 / (1.0 + Math.exp(-x)) : Math.exp(x) / (1.0 + Math.exp(x));
	}
	private static double logSigmoid(double x) {
		return x >= 0.0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x));
	}
}
