/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Candidate-specific or common Gaussian birth proposal. */
public final class GaussianRjBirthProposal implements RjBirthProposal {
	private final double[] means, standardDeviations;
	public GaussianRjBirthProposal(double mean, double standardDeviation) { this(new double[] {mean}, new double[] {standardDeviation}); }
	public GaussianRjBirthProposal(double[] means, double[] standardDeviations) {
		if (means == null || standardDeviations == null || means.length == 0 || means.length != standardDeviations.length)
			throw new IllegalArgumentException("matching Gaussian proposal parameters required");
		this.means = means.clone(); this.standardDeviations = standardDeviations.clone();
		for (int i = 0; i < this.means.length; i++) if (!Double.isFinite(this.means[i]) || !(this.standardDeviations[i] > 0.0)
				|| !Double.isFinite(this.standardDeviations[i])) throw new IllegalArgumentException("finite means and positive scales required");
	}
	@Override public double sample(int candidate, ReversibleJumpState state, RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine required"); int index = index(candidate);
		return means[index] + standardDeviations[index] * random.nextGaussian();
	}
	@Override public double logDensity(double value, int candidate, ReversibleJumpState state) {
		int index = index(candidate); double standardized = (value - means[index]) / standardDeviations[index];
		return -0.5 * standardized * standardized - Math.log(standardDeviations[index]) - 0.5 * Math.log(2.0 * Math.PI);
	}
	private int index(int candidate) {
		if (candidate < 0 || means.length != 1 && candidate >= means.length) throw new IllegalArgumentException("candidate has no Gaussian proposal");
		return means.length == 1 ? 0 : candidate;
	}
}
