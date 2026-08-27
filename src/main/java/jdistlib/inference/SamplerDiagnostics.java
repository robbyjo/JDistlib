/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Cross-chain sampler health summary. */
public final class SamplerDiagnostics {
	private final double meanAcceptanceProbability;
	private final int divergences;
	private final int treeDepthSaturations;
	private final int maximumTreeDepth;
	private final double energyBayesianFractionMissingInformation;
	private final int numericalFailures;

	SamplerDiagnostics(double meanAcceptanceProbability, int divergences,
			int treeDepthSaturations, int maximumTreeDepth,
			double energyBayesianFractionMissingInformation, int numericalFailures) {
		this.meanAcceptanceProbability = meanAcceptanceProbability;
		this.divergences = divergences;
		this.treeDepthSaturations = treeDepthSaturations;
		this.maximumTreeDepth = maximumTreeDepth;
		this.energyBayesianFractionMissingInformation = energyBayesianFractionMissingInformation;
		this.numericalFailures = numericalFailures;
	}
	public double meanAcceptanceProbability() { return meanAcceptanceProbability; }
	public int divergences() { return divergences; }
	public int treeDepthSaturations() { return treeDepthSaturations; }
	public int maximumTreeDepth() { return maximumTreeDepth; }
	public double energyBayesianFractionMissingInformation() {
		return energyBayesianFractionMissingInformation;
	}
	public int numericalFailures() { return numericalFailures; }
	public boolean healthy() {
		return divergences == 0 && treeDepthSaturations == 0
				&& numericalFailures == 0
				&& (Double.isNaN(energyBayesianFractionMissingInformation)
				|| energyBayesianFractionMissingInformation >= 0.3);
	}
}
