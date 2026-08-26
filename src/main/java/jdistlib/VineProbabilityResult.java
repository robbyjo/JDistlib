/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Monte Carlo lower-orthant probability returned by a vine copula. */
public final class VineProbabilityResult {
	public final double probability;
	public final double standardError;
	public final int samples;

	VineProbabilityResult(double probability, double standardError, int samples) {
		this.probability = probability;
		this.standardError = standardError;
		this.samples = samples;
	}
}
