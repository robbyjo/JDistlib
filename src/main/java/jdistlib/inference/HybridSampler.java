/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Scheduled support-aware sampler for fixed-dimensional mixed continuous/discrete targets. */
public final class HybridSampler implements Sampler {
	private final MixedStateSpace space; private final HybridKernel[] schedule;
	public HybridSampler(MixedStateSpace space, HybridKernel... schedule) {
		if (space == null || schedule == null || schedule.length == 0) throw new IllegalArgumentException("state space and schedule required");
		this.space = space; this.schedule = schedule.clone();
		for (HybridKernel kernel : this.schedule) if (kernel == null) throw new IllegalArgumentException("schedule must not contain null");
	}
	@Override public ChainResult sample(LogDensity target, double[] initialState, SamplingOptions options, RandomEngine random) {
		return sampleMixed(target, initialState, options, random).chain();
	}
	public HybridSamplingResult sampleMixed(LogDensity target, double[] initialState, SamplingOptions options, RandomEngine random) {
		if (target == null || options == null || random == null || !space.contains(initialState)) throw new IllegalArgumentException("valid target, state, options and random engine required");
		double[] state = initialState.clone(); double logDensity = target.logDensity(state); ChainAccumulator accumulator = new ChainAccumulator();
		String[] names = new String[schedule.length]; long[] attempts = new long[schedule.length], accepts = new long[schedule.length], supportRejections = new long[schedule.length];
		for (int i = 0; i < schedule.length; i++) names[i] = schedule[i].name();
		if (!Double.isFinite(logDensity)) {
			accumulator.warn("initial state has non-finite log density");
			return result(accumulator, state, logDensity, 0, random, options, names, attempts, accepts, supportRejections, ChainResult.Status.INVALID_INITIAL_STATE);
		}
		int total = options.warmupIterations() + options.sampleIterations() * options.thinning(), completed = 0; double warmupAcceptance = 0.0;
		for (int iteration = 0; iteration < total; iteration++) {
			boolean warmup = iteration < options.warmupIterations(); boolean anyAccepted = false; double acceptance = 0.0;
			for (int kernel = 0; kernel < schedule.length; kernel++) {
				HybridKernelTransition transition = schedule[kernel].update(state, logDensity, target, space, random);
				attempts[kernel]++; if (transition.accepted()) { accepts[kernel]++; anyAccepted = true; }
				if (transition.supportRejected()) supportRejections[kernel]++;
				acceptance += transition.acceptanceProbability(); logDensity = transition.logDensity();
			}
			acceptance /= schedule.length; if (warmup) warmupAcceptance += acceptance; completed++;
			IterationStats stats = new IterationStats(anyAccepted, acceptance, options.stepSize(), -logDensity, 0.0, false, 0, schedule.length);
			options.progress(completed, total, warmup, stats);
			if (options.cancelled()) return result(accumulator, state, logDensity, completed, random, options, names, attempts, accepts, supportRejections, ChainResult.Status.CANCELLED);
			if (!warmup && (iteration - options.warmupIterations()) % options.thinning() == 0) accumulator.retain(options, state, logDensity, stats);
		}
		WarmupResult warmup = new WarmupResult(options.warmupIterations(), options.stepSize(), options.stepSize(), null,
				options.warmupIterations() == 0 ? Double.NaN : warmupAcceptance / options.warmupIterations());
		ChainResult chain = accumulator.result(state, logDensity, completed, random, warmup, ChainResult.Status.SUCCESS);
		return new HybridSamplingResult(chain, new HybridSamplerDiagnostics(names, attempts, accepts, supportRejections));
	}
	private HybridSamplingResult result(ChainAccumulator accumulator, double[] state, double logDensity, int completed,
			RandomEngine random, SamplingOptions options, String[] names, long[] attempts, long[] accepts,
			long[] supportRejections, ChainResult.Status status) {
		WarmupResult warmup = new WarmupResult(Math.min(completed, options.warmupIterations()), options.stepSize(), options.stepSize(), null, Double.NaN);
		return new HybridSamplingResult(accumulator.result(state, logDensity, completed, random, warmup, status),
				new HybridSamplerDiagnostics(names, attempts, accepts, supportRejections));
	}
}
