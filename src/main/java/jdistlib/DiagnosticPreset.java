/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.math.IntegrationOptions;

/** Balanced starting points for custom-kernel diagnostics. */
public enum DiagnosticPreset {
	FAST(33, 16, 2, 100, 100000, 1e-8),
	STANDARD(257, 128, 4, 300, 250000, 1e-9),
	THOROUGH(1025, 1024, 8, 1000, 1000000, 1e-11);

	private final int sampleCount;
	private final int randomBudget;
	private final int adaptiveRounds;
	private final int subdivisions;
	private final int evaluations;
	private final double tolerance;

	DiagnosticPreset(int sampleCount, int randomBudget, int adaptiveRounds,
			int subdivisions, int evaluations, double tolerance) {
		this.sampleCount = sampleCount;
		this.randomBudget = randomBudget;
		this.adaptiveRounds = adaptiveRounds;
		this.subdivisions = subdivisions;
		this.evaluations = evaluations;
		this.tolerance = tolerance;
	}

	/** Creates independently editable settings for this preset. */
	public FunctionAnalysisOptions options() {
		return FunctionAnalysisOptions.builder()
				.sampleCount(sampleCount)
				.randomizedProbeBudget(randomBudget)
				.adaptiveProbeRounds(adaptiveRounds)
				.integrationOptions(IntegrationOptions.builder()
						.tolerances(tolerance, tolerance)
						.subdivisions(subdivisions)
						.maxEvaluations(evaluations)
						.method(IntegrationOptions.Method.AUTO)
						.build())
				.build();
	}
}
