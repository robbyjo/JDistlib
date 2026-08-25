/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Numerical self-consistency checks for a constructed distribution. */
public final class DistributionAnalysis {
	private final List<DiagnosticFinding> findings;
	private final double normalizationRelativeError;
	private final double maximumTailDisagreement;
	private final double maximumQuantileRoundTripError;
	private final double mean;
	private final double variance;
	private final double firstAbsoluteMoment;
	private final double secondAbsoluteMoment;
	private final boolean momentsStable;
	private final List<AbsoluteMomentAnalysis> absoluteMoments;

	DistributionAnalysis(List<DiagnosticFinding> findings,
			double normalizationRelativeError, double maximumTailDisagreement,
			double maximumQuantileRoundTripError, double mean, double variance,
			double firstAbsoluteMoment, double secondAbsoluteMoment,
			boolean momentsStable, List<AbsoluteMomentAnalysis> absoluteMoments) {
		this.findings = Collections.unmodifiableList(
				new ArrayList<DiagnosticFinding>(findings));
		this.normalizationRelativeError = normalizationRelativeError;
		this.maximumTailDisagreement = maximumTailDisagreement;
		this.maximumQuantileRoundTripError = maximumQuantileRoundTripError;
		this.mean = mean;
		this.variance = variance;
		this.firstAbsoluteMoment = firstAbsoluteMoment;
		this.secondAbsoluteMoment = secondAbsoluteMoment;
		this.momentsStable = momentsStable;
		this.absoluteMoments = Collections.unmodifiableList(
				new ArrayList<AbsoluteMomentAnalysis>(absoluteMoments));
	}

	public List<DiagnosticFinding> getFindings() { return findings; }
	public double getNormalizationRelativeError() {
		return normalizationRelativeError;
	}
	public double getMaximumTailDisagreement() { return maximumTailDisagreement; }
	public double getMaximumQuantileRoundTripError() {
		return maximumQuantileRoundTripError;
	}
	public double getMean() { return mean; }
	public double getVariance() { return variance; }
	/** Returns the estimated first absolute moment, E[|X|]. */
	public double getFirstAbsoluteMoment() { return firstAbsoluteMoment; }
	/** Returns the estimated second absolute moment, E[|X|^2]. */
	public double getSecondAbsoluteMoment() { return secondAbsoluteMoment; }
	public boolean areMomentsStable() { return momentsStable; }
	public List<AbsoluteMomentAnalysis> getAbsoluteMoments() {
		return absoluteMoments;
	}

	/** Returns the requested order's report, or null when it was not requested. */
	public AbsoluteMomentAnalysis getAbsoluteMoment(double order) {
		for (AbsoluteMomentAnalysis moment : absoluteMoments) {
			if (moment.getOrder() == order) return moment;
		}
		return null;
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

	/** Returns a versioned RFC 8259 JSON diagnostic report. */
	public String toJson() { return DiagnosticJson.toJson(this); }
}
