/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Final adjusted-MCLMC chain and auditable pilot objective values. */
public final class AdjustedMclmcTuningResult {
	private final ChainResult chain; private final int leapfrogSteps; private final double stepSize;
	private final int[] candidates; private final double[] scores, acceptances;
	private final double[] massScaling;
	AdjustedMclmcTuningResult(ChainResult chain, int leapfrogSteps, double stepSize, int[] candidates, double[] scores, double[] acceptances) {
		this(chain, leapfrogSteps, stepSize, candidates, scores, acceptances, ones(chain.dimension()));
	}
	AdjustedMclmcTuningResult(ChainResult chain, int leapfrogSteps, double stepSize, int[] candidates, double[] scores, double[] acceptances, double[] massScaling) {
		this.chain = chain; this.leapfrogSteps = leapfrogSteps; this.stepSize = stepSize; this.candidates = candidates.clone(); this.scores = scores.clone(); this.acceptances = acceptances.clone(); this.massScaling = massScaling.clone(); }
	public ChainResult chain() { return chain; } public int leapfrogSteps() { return leapfrogSteps; } public double stepSize() { return stepSize; }
	public int[] candidates() { return candidates.clone(); } public double[] scores() { return scores.clone(); } public double[] acceptanceProbabilities() { return acceptances.clone(); }
	public double[] massScaling() { return massScaling.clone(); }
	private static double[] ones(int dimension) { double[] result = new double[dimension]; for (int i = 0; i < dimension; i++) result[i] = 1.0; return result; }
}
