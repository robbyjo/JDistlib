/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.rng.RandomEngine;

/**
 * Locally informed candidate proposal using a prepared {@code X'v} product.
 * A uniform mixture component preserves support for every inactive candidate.
 */
public final class ResidualInformedSparseCandidateProposal implements SparseCandidateProposal {
	private final PreparedTransposeProduct product; private final SparseResidualProvider residuals;
	private final double temperature, uniformMixture;
	public ResidualInformedSparseCandidateProposal(PreparedTransposeProduct product,
			SparseResidualProvider residuals, double temperature, double uniformMixture) {
		if (product == null || residuals == null || !(temperature > 0.0) || !Double.isFinite(temperature)
				|| !(uniformMixture > 0.0 && uniformMixture <= 1.0)) throw new IllegalArgumentException("prepared scores, temperature, and uniform mixture required");
		this.product = product; this.residuals = residuals; this.temperature = temperature; this.uniformMixture = uniformMixture;
	}
	@Override public SparseCandidateChoice sample(SparseSubsetState state, SparseSubsetTarget target, RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine required"); double[] probabilities = probabilities(state, target);
		double threshold = random.nextDouble(), cumulative = 0.0; int last = -1;
		for (int candidate = 0; candidate < probabilities.length; candidate++) if (probabilities[candidate] > 0.0) {
			last = candidate; cumulative += probabilities[candidate]; if (threshold <= cumulative) return new SparseCandidateChoice(candidate, Math.log(probabilities[candidate]));
		}
		return new SparseCandidateChoice(last, Math.log(probabilities[last]));
	}
	@Override public double logProbability(int candidate, SparseSubsetState state, SparseSubsetTarget target) {
		if (candidate < 0 || candidate >= target.candidateCount() || state.active(candidate)) return Double.NEGATIVE_INFINITY;
		return Math.log(probabilities(state, target)[candidate]);
	}
	private double[] probabilities(SparseSubsetState state, SparseSubsetTarget target) {
		if (state == null || target == null) throw new IllegalArgumentException("state and target required"); target.validate(state);
		if (product.columns() != target.candidateCount()) throw new IllegalArgumentException("prepared score columns do not match candidates");
		double[] scoreVector = residuals.scoreVector(state);
		if (scoreVector == null || scoreVector.length != product.rows()) throw new IllegalArgumentException("score vector rows do not match prepared matrix");
		double[] scores = product.multiply(new double[][] {scoreVector})[0]; int inactive = target.candidateCount() - state.size();
		if (inactive == 0) throw new IllegalArgumentException("full sparse model has no inactive candidate");
		double maximum = Double.NEGATIVE_INFINITY;
		for (int candidate = 0; candidate < scores.length; candidate++) if (!state.active(candidate)) maximum = Math.max(maximum, temperature * Math.abs(scores[candidate]));
		double total = 0.0; double[] probabilities = new double[scores.length];
		for (int candidate = 0; candidate < scores.length; candidate++) if (!state.active(candidate)) { double value = Math.exp(temperature * Math.abs(scores[candidate]) - maximum); probabilities[candidate] = value; total += value; }
		double uniform = uniformMixture / inactive, informed = (1.0 - uniformMixture) / total;
		for (int candidate = 0; candidate < probabilities.length; candidate++) if (!state.active(candidate)) probabilities[candidate] = uniform + informed * probabilities[candidate];
		return probabilities;
	}
	@Override public void close() { product.close(); }
}
