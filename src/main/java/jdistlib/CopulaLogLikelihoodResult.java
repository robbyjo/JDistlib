/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Auditable likelihood aggregation for continuous, discrete, or mixed data. */
public final class CopulaLogLikelihoodResult {
	public enum Status {
		SUCCESS,
		NUMERICAL_WARNING,
		ZERO_LIKELIHOOD,
		INVALID_INPUT,
		MEASURE_FAILURE
	}

	private final double logLikelihood;
	private final CopulaMeasureResult[] contributions;
	private final int successfulContributions;
	private final int warningContributions;
	private final int zeroContributions;
	private final long cdfEvaluations;
	private final double maximumAbsoluteError;
	private final int firstProblemIndex;
	private final Status status;
	private final String message;

	CopulaLogLikelihoodResult(double logLikelihood,
			CopulaMeasureResult[] contributions, int successfulContributions,
			int warningContributions, int zeroContributions, long cdfEvaluations,
			double maximumAbsoluteError, int firstProblemIndex, Status status,
			String message) {
		this.logLikelihood = logLikelihood;
		this.contributions = contributions.clone();
		this.successfulContributions = successfulContributions;
		this.warningContributions = warningContributions;
		this.zeroContributions = zeroContributions;
		this.cdfEvaluations = cdfEvaluations;
		this.maximumAbsoluteError = maximumAbsoluteError;
		this.firstProblemIndex = firstProblemIndex;
		this.status = status;
		this.message = message;
	}

	public double getLogLikelihood() { return logLikelihood; }
	public int getObservations() { return contributions.length; }
	public int getSuccessfulContributions() { return successfulContributions; }
	public int getWarningContributions() { return warningContributions; }
	public int getZeroContributions() { return zeroContributions; }
	public long getCdfEvaluations() { return cdfEvaluations; }
	public double getMaximumAbsoluteError() { return maximumAbsoluteError; }
	public int getFirstProblemIndex() { return firstProblemIndex; }
	public CopulaMeasureResult[] getContributions() { return contributions.clone(); }
	public Status getStatus() { return status; }
	public String getMessage() { return message; }
	public String message() { return message; }
	public boolean isSuccess() { return status == Status.SUCCESS; }
	public boolean hasEstimate() { return !Double.isNaN(logLikelihood); }
}
