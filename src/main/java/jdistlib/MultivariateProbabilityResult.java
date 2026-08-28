/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Result of a numerical multivariate probability calculation. */
public final class MultivariateProbabilityResult {
	/** Legacy integer code corresponding to {@link MultivariateProbabilityStatus#SUCCESS}. */
	public static final int SUCCESS = 0;
	/** Legacy integer code corresponding to {@link MultivariateProbabilityStatus#MAX_EVALUATIONS_REACHED}. */
	public static final int MAX_EVALUATIONS_REACHED = 1;
	/** Legacy integer code corresponding to {@link MultivariateProbabilityStatus#INVALID_INPUT}. */
	public static final int INVALID_INPUT = 2;

	/** The requested probability. */
	public final double probability;
	/**
	 * Replication-based absolute error indicator. This is a conservative
	 * convergence heuristic, not a rigorous confidence bound.
	 */
	public final double absoluteError;
	/** Number of transformed-integrand evaluations. */
	public final int evaluations;
	/**
	 * Stable legacy status code. New code should prefer {@link #getStatus()}.
	 */
	public final int status;
	private final MultivariateProbabilityStatus typedStatus;

	MultivariateProbabilityResult(double probability, double absoluteError,
			int evaluations, int status) {
		this.probability = probability;
		this.absoluteError = absoluteError;
		this.evaluations = evaluations;
		this.status = status;
		this.typedStatus = MultivariateProbabilityStatus.fromCode(status);
	}

	/** Returns whether the requested numerical tolerance was met. */
	public boolean isSuccess() {
		return typedStatus == MultivariateProbabilityStatus.SUCCESS;
	}

	/** Alias emphasizing convergence rather than validity. */
	public boolean isConverged() {
		return isSuccess();
	}

	/** Returns whether this object contains a usable numerical estimate. */
	public boolean hasEstimate() {
		return typedStatus != MultivariateProbabilityStatus.INVALID_INPUT &&
				Double.isFinite(probability);
	}

	/** Returns the typed terminal status. */
	public MultivariateProbabilityStatus getStatus() {
		return typedStatus;
	}

	public String message() {
		return typedStatus.message();
	}
}
