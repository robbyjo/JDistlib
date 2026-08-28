/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Deterministic many-short-chain execution using common-initialization superchains. */
public final class ManyShortChains {
	private ManyShortChains() {}
	public static ManyShortChainsResult run(Sampler sampler, LogDensity target,
			double[][] superchainInitialStates, int chainsPerSuperchain,
			SamplingOptions options, long baseSeed, int parallelism) {
		return run(sampler, target, new SuperchainPlan(superchainInitialStates, chainsPerSuperchain), options, baseSeed, parallelism);
	}
	public static ManyShortChainsResult run(Sampler sampler, LogDensity target, SuperchainPlan plan,
			SamplingOptions options, long baseSeed, int parallelism) { if (plan == null) throw new IllegalArgumentException("superchain plan is required");
		int[] ids = plan.ids(); return new ManyShortChainsResult(Chains.parallel(sampler, target, plan.expandedStates(), options, baseSeed, parallelism), ids, plan); }
}
