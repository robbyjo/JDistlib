/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Results and nested convergence diagnostics from a superchain design. */
public final class ManyShortChainsResult {
	private final ChainResult[] chains; private final int[] superchainIds;
	private final SuperchainPlan plan;
	ManyShortChainsResult(ChainResult[] chains, int[] superchainIds, SuperchainPlan plan) { this.chains = chains.clone(); this.superchainIds = superchainIds.clone(); this.plan = plan; }
	public ChainResult[] chains() { return chains.clone(); }
	public int[] superchainIds() { return superchainIds.clone(); }
	public SuperchainPlan plan() { return plan; }
	public double nestedRHat(int coordinate) { return McmcDiagnostics.nestedRHat(values(coordinate), superchainIds); }
	public double nestedRankNormalizedRHat(int coordinate) { return McmcDiagnostics.nestedRankNormalizedRHat(values(coordinate), superchainIds); }
	private double[][] values(int coordinate) { double[][] result = new double[chains.length][];
		for (int chain = 0; chain < chains.length; chain++) { if (coordinate < 0 || coordinate >= chains[chain].dimension()) throw new IllegalArgumentException("coordinate out of range");
			result[chain] = new double[chains[chain].size()]; for (int draw = 0; draw < result[chain].length; draw++) result[chain][draw] = chains[chain].valueAt(draw, coordinate); } return result; }
}
