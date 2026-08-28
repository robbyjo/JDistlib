/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Concise facade for reproducible multi-chain fitting. */
public final class Inference {
	private Inference() {}
	public static Fit fit(Sampler sampler, LogDensity target, double[][] initialStates,
			SamplingOptions options, long seed, int parallelism) {
		String identity = target.getClass().getName() + ":" + initialStates[0].length;
		return fit(sampler, target, identity, initialStates, options, seed, parallelism);
	}
	/** Fits with a stable caller-provided model/source identity included in the manifest hash. */
	public static Fit fit(Sampler sampler, LogDensity target, String modelIdentity,
			double[][] initialStates, SamplingOptions options, long seed, int parallelism) {
		long startedMillis = System.currentTimeMillis();
		long startedNanos = System.nanoTime();
		ChainResult[] chains = Chains.parallel(sampler, target, initialStates, options,
				seed, parallelism);
		long elapsed = System.nanoTime() - startedNanos;
		return new Fit(chains, McmcDiagnostics.analyze(chains), RunManifest.create(
				sampler, target, modelIdentity, options, seed, startedMillis, elapsed));
	}
}
