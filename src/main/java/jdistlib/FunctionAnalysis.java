/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import jdistlib.math.IntegrationStabilityResult;

/** Evidence gathered while probing and repeatedly integrating a kernel. */
public final class FunctionAnalysis {
	private final List<DiagnosticFinding> findings;
	private final int sampledPoints;
	private final int randomizedSampledPoints;
	private final long randomSeed;
	private final double minimumPositiveValue;
	private final double maximumValue;
	private final double[] suggestedBreakpoints;
	private final IntegrationStabilityResult normalizationStability;

	FunctionAnalysis(List<DiagnosticFinding> findings, int sampledPoints,
			int randomizedSampledPoints, long randomSeed,
			double minimumPositiveValue, double maximumValue,
			double[] suggestedBreakpoints,
			IntegrationStabilityResult normalizationStability) {
		this.findings = Collections.unmodifiableList(
				new ArrayList<DiagnosticFinding>(findings));
		this.sampledPoints = sampledPoints;
		this.randomizedSampledPoints = randomizedSampledPoints;
		this.randomSeed = randomSeed;
		this.minimumPositiveValue = minimumPositiveValue;
		this.maximumValue = maximumValue;
		this.suggestedBreakpoints = suggestedBreakpoints.clone();
		this.normalizationStability = normalizationStability;
	}

	public List<DiagnosticFinding> getFindings() { return findings; }
	public int getSampledPoints() { return sampledPoints; }
	public int getRandomizedSampledPoints() { return randomizedSampledPoints; }
	public long getRandomSeed() { return randomSeed; }
	public double getMinimumPositiveValue() { return minimumPositiveValue; }
	public double getMaximumValue() { return maximumValue; }
	public double[] getSuggestedBreakpoints() { return suggestedBreakpoints.clone(); }
	public IntegrationStabilityResult getNormalizationStability() {
		return normalizationStability;
	}

	public boolean hasErrors() {
		for (DiagnosticFinding finding : findings) {
			if (finding.getSeverity() == DiagnosticFinding.Severity.ERROR) return true;
		}
		return false;
	}

	public boolean hasWarnings() {
		for (DiagnosticFinding finding : findings) {
			if (finding.getSeverity() == DiagnosticFinding.Severity.WARNING) return true;
		}
		return false;
	}

	/** True means no error was observed; it is not a mathematical proof. */
	public boolean isSuitableForConstruction() { return !hasErrors(); }

	/** Applies an explicit policy to the retained findings. */
	public boolean isSuitableForConstruction(ConstructionPolicy policy) {
		if (policy == null) throw new IllegalArgumentException("policy must not be null");
		if (policy == ConstructionPolicy.PERMISSIVE) return true;
		return policy == ConstructionPolicy.STRICT
				? !hasErrors() && !hasWarnings() : !hasErrors();
	}

	/** Returns a versioned RFC 8259 JSON diagnostic report. */
	public String toJson() { return DiagnosticJson.toJson(this); }
}
