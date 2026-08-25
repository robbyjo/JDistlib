/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Result of a numerical multivariate rectangle-probability calculation. */
public final class MultivariateProbabilityResult {
	public static final int SUCCESS = 0;
	public static final int MAX_EVALUATIONS_REACHED = 1;
	public static final int INVALID_INPUT = 2;

	/** The requested probability. */
	public final double probability;
	/** Estimated absolute numerical error. */
	public final double absoluteError;
	/** Number of transformed-integrand evaluations. */
	public final int evaluations;
	/** 0 for success, 1 for the evaluation limit, and 2 for invalid input. */
	public final int status;

	MultivariateProbabilityResult(double probability, double absoluteError,
			int evaluations, int status) {
		this.probability = probability;
		this.absoluteError = absoluteError;
		this.evaluations = evaluations;
		this.status = status;
	}

	public boolean isSuccess() {
		return status == SUCCESS;
	}

	public String message() {
		switch (status) {
		case 0: return "OK";
		case 1: return "maximum number of evaluations reached";
		case 2: return "invalid parameters or bounds";
		default: return "unknown multivariate probability status";
		}
	}
}
