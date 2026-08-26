/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Joint density, mass, or mixed-measure likelihood contribution. */
public final class CopulaMeasureResult {
	public enum Status {
		EXACT_CONTINUOUS,
		RECTANGLE_DIFFERENCE,
		NUMERICAL_MIXED_DERIVATIVE,
		NUMERICAL_WARNING,
		INVALID_INPUT,
		EVALUATION_BUDGET_EXCEEDED
	}

	public final double value;
	public final double logValue;
	public final double absoluteError;
	public final int cdfEvaluations;
	private final Status status;
	private final String message;

	CopulaMeasureResult(double value, double absoluteError, int evaluations,
			Status status, String message) {
		this(value, value == 0.0 ? Double.NEGATIVE_INFINITY : Math.log(value),
				absoluteError, evaluations, status, message);
	}

	CopulaMeasureResult(double value, double logValue, double absoluteError,
			int evaluations, Status status, String message) {
		this.value = value;
		this.logValue = logValue;
		this.absoluteError = absoluteError;
		this.cdfEvaluations = evaluations;
		this.status = status;
		this.message = message;
	}

	public Status getStatus() { return status; }
	public String message() { return message; }
	public boolean isSuccess() {
		return status == Status.EXACT_CONTINUOUS
				|| status == Status.RECTANGLE_DIFFERENCE
				|| status == Status.NUMERICAL_MIXED_DERIVATIVE;
	}
	public boolean hasEstimate() { return !Double.isNaN(logValue); }
}
