/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Automatic adjusted-MCLMC pilot search for integrator step and decorrelation length. */
public final class AdjustedMclmcTuner {
	private AdjustedMclmcTuner() {}
	public static AdjustedMclmcTuningResult tuneAndSample(LogDensity target, double[] initialState,
			SamplingOptions sampling, AdjustedMclmcTuningOptions tuning, RandomEngine random) {
		if (target == null || initialState == null || sampling == null || tuning == null || random == null)
			throw new IllegalArgumentException("target, state, options, and random are required");
		AdjustedMicrocanonicalLangevin sampler = new AdjustedMicrocanonicalLangevin();
		int[] candidates = candidates(tuning.maximumLeapfrogSteps()); double[] scores = new double[candidates.length];
		double[] acceptances = new double[candidates.length], steps = new double[candidates.length]; double[][] scales = new double[candidates.length][];
		int selected = 0; double best = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < candidates.length; i++) {
			SamplingOptions pilotOptions = sampling.toBuilder().warmupIterations(tuning.pilotWarmup())
					.sampleIterations(tuning.pilotDraws()).thinning(1).leapfrogSteps(candidates[i])
					.targetAcceptance(tuning.targetAcceptance()).storeDraws(true).drawSink(null).progressListener(null).build();
			ChainResult pilot = sampler.sample(target, initialState, pilotOptions, random);
			steps[i] = pilot.warmup().finalStepSize(); acceptances[i] = meanAcceptance(pilot);
			scales[i] = marginalScales(pilot);
			scores[i] = jumpScore(pilot, candidates[i]) * Math.exp(-5.0 * Math.abs(acceptances[i] - tuning.targetAcceptance()));
			if (scores[i] > best) { best = scores[i]; selected = i; }
		}
		SamplingOptions finalOptions = sampling.toBuilder().leapfrogSteps(candidates[selected]).stepSize(steps[selected]).build();
		DifferentiableLogDensity differentiable = HamiltonianSupport.gradientTarget(target, finalOptions);
		final double[] selectedScale = scales[selected]; final double logJacobian = logProduct(selectedScale);
		DifferentiableLogDensity scaledTarget = new DifferentiableLogDensity() {
			@Override public double logDensityAndGradient(double[] state, double[] gradient) { double[] original = new double[state.length];
				for (int i = 0; i < state.length; i++) original[i] = state[i] * selectedScale[i]; double[] originalGradient = new double[state.length];
				double value = differentiable.logDensityAndGradient(original, originalGradient); for (int i = 0; i < state.length; i++) gradient[i] = originalGradient[i] * selectedScale[i]; return value + logJacobian; }
		};
		double[] scaledInitial = new double[initialState.length]; for (int i = 0; i < initialState.length; i++) scaledInitial[i] = initialState[i] / selectedScale[i];
		ChainResult scaled = sampler.sample(scaledTarget, scaledInitial, finalOptions, random); ChainResult result = unscale(scaled, selectedScale, logJacobian);
		return new AdjustedMclmcTuningResult(result, candidates[selected], result.warmup().finalStepSize(), candidates, scores, acceptances, selectedScale);
	}
	private static int[] candidates(int maximum) { List<Integer> values = new ArrayList<Integer>(); for (int value = 1; value < maximum; value *= 2) values.add(value);
		if (values.isEmpty() || values.get(values.size() - 1) != maximum) values.add(maximum); int[] result = new int[values.size()]; for (int i = 0; i < result.length; i++) result[i] = values.get(i); return result; }
	private static double meanAcceptance(ChainResult chain) { double result = 0.0; for (int i = 0; i < chain.size(); i++) result += chain.statisticsAt(i).acceptanceProbability(); return result / chain.size(); }
	private static double jumpScore(ChainResult chain, int cost) { if (chain.size() < 2) return 0.0; double result = 0.0;
		for (int draw = 1; draw < chain.size(); draw++) { double[] previous = chain.sample(draw - 1), current = chain.sample(draw); for (int d = 0; d < current.length; d++) { double difference = current[d] - previous[d]; result += difference * difference; } }
		return result / ((chain.size() - 1.0) * cost); }
	private static double[] marginalScales(ChainResult chain) { int dimension = chain.dimension(); double[] mean = new double[dimension], result = new double[dimension];
		for (int draw = 0; draw < chain.size(); draw++) for (int d = 0; d < dimension; d++) mean[d] += chain.valueAt(draw, d); for (int d = 0; d < dimension; d++) mean[d] /= chain.size();
		for (int draw = 0; draw < chain.size(); draw++) for (int d = 0; d < dimension; d++) { double difference = chain.valueAt(draw, d) - mean[d]; result[d] += difference * difference; }
		for (int d = 0; d < dimension; d++) result[d] = Math.max(0.1, Math.min(10.0, Math.sqrt(result[d] / Math.max(1.0, chain.size() - 1.0)))); return result; }
	private static double logProduct(double[] values) { double result = 0.0; for (double value : values) result += Math.log(value); return result; }
	private static ChainResult unscale(ChainResult source, double[] scale, double logJacobian) { double[][] samples = source.samples(); for (double[] sample : samples) for (int d = 0; d < scale.length; d++) sample[d] *= scale[d];
		double[] densities = source.logDensities(); for (int i = 0; i < densities.length; i++) densities[i] -= logJacobian; ChainCheckpoint checkpoint = source.checkpoint(); double[] state = checkpoint.state(); for (int d = 0; d < scale.length; d++) state[d] *= scale[d];
		ChainCheckpoint transformed = new ChainCheckpoint(state, checkpoint.logDensity() - logJacobian, checkpoint.completedIterations(), checkpoint.random(), checkpoint.samplerCheckpoint());
		return new ChainResult(samples, densities, source.statistics(), source.warmup(), transformed, source.status(), source.warnings()); }
}
