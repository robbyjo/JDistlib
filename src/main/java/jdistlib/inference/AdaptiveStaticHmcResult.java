/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Draws plus the shared trajectory adaptation selected by ChEES or SNAPER. */
public final class AdaptiveStaticHmcResult {
	private final ChainResult[] chains; private final int leapfrogSteps; private final double stepSize;
	private final double[] principalDirection, candidateScores;
	AdaptiveStaticHmcResult(ChainResult[] chains, int leapfrogSteps, double stepSize, double[] principalDirection, double[] candidateScores) {
		this.chains = chains.clone(); this.leapfrogSteps = leapfrogSteps; this.stepSize = stepSize;
		this.principalDirection = principalDirection.clone(); this.candidateScores = candidateScores.clone(); }
	public ChainResult[] chains() { return chains.clone(); } public int leapfrogSteps() { return leapfrogSteps; }
	public double stepSize() { return stepSize; } public double integrationTime() { return leapfrogSteps * stepSize; }
	public double[] principalDirection() { return principalDirection.clone(); } public double[] candidateScores() { return candidateScores.clone(); }
}
