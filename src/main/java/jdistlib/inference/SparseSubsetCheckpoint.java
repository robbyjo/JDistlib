/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Exact sparse RJ restart state, adaptation, counters, RNG, and online summaries. */
public final class SparseSubsetCheckpoint {
	private final SparseSubsetState state; private final double logJoint; private final long completedTransitions, retainedDraws;
	private final RandomEngine random; private final String[] moveNames; private final double[] moveWeights, logScales;
	private final long[] scaleUpdates, moveAttempts, moveAccepts, invalidProposals, modelSizeCounts, inclusionCounts, coefficientCounts;
	private final double[] coefficientSums, coefficientSquareSums, commonSums, commonSquareSums;
	private final long warmupIterations; private final boolean warmupComplete;
	public SparseSubsetCheckpoint(SparseSubsetState state, double logJoint, long completedTransitions, long retainedDraws,
			RandomEngine random, String[] moveNames, double[] moveWeights, double[] logScales, long[] scaleUpdates,
			long[] moveAttempts, long[] moveAccepts, long[] invalidProposals, long[] modelSizeCounts,
			long[] inclusionCounts, long[] coefficientCounts, double[] coefficientSums, double[] coefficientSquareSums,
			double[] commonSums, double[] commonSquareSums, long warmupIterations, boolean warmupComplete) {
		if (state == null || Double.isNaN(logJoint) || completedTransitions < 0L || retainedDraws < 0L || random == null
				|| warmupIterations < 0L || warmupComplete != (completedTransitions >= warmupIterations)
				|| moveNames == null || moveWeights == null || moveNames.length != moveWeights.length || moveAttempts == null
				|| moveAccepts == null || invalidProposals == null || moveNames.length != moveAttempts.length
				|| moveNames.length != moveAccepts.length || moveNames.length != invalidProposals.length || logScales == null
				|| scaleUpdates == null || logScales.length != scaleUpdates.length || modelSizeCounts == null
				|| modelSizeCounts.length != logScales.length || inclusionCounts == null || coefficientCounts == null
				|| coefficientSums == null || coefficientSquareSums == null || inclusionCounts.length != coefficientCounts.length
				|| inclusionCounts.length != coefficientSums.length || inclusionCounts.length != coefficientSquareSums.length
				|| commonSums == null || commonSquareSums == null || commonSums.length != commonSquareSums.length)
			throw new IllegalArgumentException("complete sparse checkpoint state required");
		this.state = state; this.logJoint = logJoint; this.completedTransitions = completedTransitions; this.retainedDraws = retainedDraws;
		this.random = random.clone(); this.moveNames = moveNames.clone(); this.moveWeights = moveWeights.clone(); this.logScales = logScales.clone();
		this.scaleUpdates = scaleUpdates.clone(); this.moveAttempts = moveAttempts.clone(); this.moveAccepts = moveAccepts.clone();
		this.invalidProposals = invalidProposals.clone(); this.modelSizeCounts = modelSizeCounts.clone(); this.inclusionCounts = inclusionCounts.clone();
		this.coefficientCounts = coefficientCounts.clone(); this.coefficientSums = coefficientSums.clone(); this.coefficientSquareSums = coefficientSquareSums.clone();
		this.commonSums = commonSums.clone(); this.commonSquareSums = commonSquareSums.clone(); this.warmupIterations = warmupIterations; this.warmupComplete = warmupComplete;
		validate();
	}
	private void validate() {
		for (int i = 0; i < moveNames.length; i++) if (moveNames[i] == null || moveNames[i].trim().isEmpty()
				|| !(moveWeights[i] > 0.0) || !Double.isFinite(moveWeights[i]) || moveAttempts[i] < 0L
				|| moveAccepts[i] < 0L || invalidProposals[i] < 0L || moveAccepts[i] > moveAttempts[i]) throw new IllegalArgumentException("valid sparse move state required");
		for (int i = 0; i < logScales.length; i++) if (!Double.isFinite(logScales[i]) || scaleUpdates[i] < 0L || modelSizeCounts[i] < 0L) throw new IllegalArgumentException("valid sparse adaptation required");
		for (int i = 0; i < inclusionCounts.length; i++) if (inclusionCounts[i] < 0L || coefficientCounts[i] < 0L
				|| inclusionCounts[i] != coefficientCounts[i] || !Double.isFinite(coefficientSums[i]) || !Double.isFinite(coefficientSquareSums[i])) throw new IllegalArgumentException("valid sparse candidate summaries required");
		for (int i = 0; i < commonSums.length; i++) if (!Double.isFinite(commonSums[i]) || !Double.isFinite(commonSquareSums[i])) throw new IllegalArgumentException("valid common summaries required");
	}
	public SparseSubsetState state() { return state; }
	public double logJoint() { return logJoint; }
	public long completedTransitions() { return completedTransitions; }
	public long retainedDraws() { return retainedDraws; }
	public RandomEngine random() { return random.clone(); }
	public String[] moveNames() { return moveNames.clone(); }
	public double[] moveWeights() { return moveWeights.clone(); }
	public double[] logScales() { return logScales.clone(); }
	public long[] scaleUpdates() { return scaleUpdates.clone(); }
	public long[] moveAttempts() { return moveAttempts.clone(); }
	public long[] moveAccepts() { return moveAccepts.clone(); }
	public long[] invalidProposals() { return invalidProposals.clone(); }
	public long[] modelSizeCounts() { return modelSizeCounts.clone(); }
	public long[] inclusionCounts() { return inclusionCounts.clone(); }
	public long[] coefficientCounts() { return coefficientCounts.clone(); }
	public double[] coefficientSums() { return coefficientSums.clone(); }
	public double[] coefficientSquareSums() { return coefficientSquareSums.clone(); }
	public double[] commonSums() { return commonSums.clone(); }
	public double[] commonSquareSums() { return commonSquareSums.clone(); }
	public long warmupIterations() { return warmupIterations; }
	public boolean warmupComplete() { return warmupComplete; }
}
