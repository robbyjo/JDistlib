/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Analysis plus the result of attempting to construct a numerical distribution. */
public final class NumericalDistributionBuildResult {
	private final FunctionAnalysis analysis;
	private final NumericalContinuousDistribution distribution;
	private final IllegalArgumentException failure;

	NumericalDistributionBuildResult(FunctionAnalysis analysis,
			NumericalContinuousDistribution distribution,
			IllegalArgumentException failure) {
		this.analysis = analysis;
		this.distribution = distribution;
		this.failure = failure;
	}

	public FunctionAnalysis getAnalysis() { return analysis; }
	public boolean canBuild() { return distribution != null; }
	public IllegalArgumentException getFailure() { return failure; }

	/** Returns the constructed distribution or throws with the retained cause. */
	public NumericalContinuousDistribution build() {
		if (distribution == null) {
			throw new IllegalStateException("the analyzed kernel could not be constructed",
					failure);
		}
		return distribution;
	}

	/** Returns the analysis and construction outcome as versioned JSON. */
	public String toJson() { return DiagnosticJson.toJson(this); }
}
