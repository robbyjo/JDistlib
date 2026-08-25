/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/**
 * Accuracy and work limits for randomized multivariate probability integration.
 * Convergence requires the replication-based error indicator to be no larger
 * than {@code max(absoluteTolerance, relativeTolerance * abs(probability))}.
 */
public final class MultivariateProbabilityOptions {
	private static final double DEFAULT_ABSOLUTE_TOLERANCE = 1e-6;
	private static final double DEFAULT_RELATIVE_TOLERANCE = 1e-5;
	private static final int DEFAULT_MAX_EVALUATIONS = 131072;
	private static final int DEFAULT_REPLICATIONS = 12;

	public final double absoluteTolerance;
	public final double relativeTolerance;
	public final int maxEvaluations;
	public final int replications;

	/** Uses 12 randomized replications and at most 131072 evaluations. */
	public MultivariateProbabilityOptions() {
		this(DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE,
				DEFAULT_MAX_EVALUATIONS, DEFAULT_REPLICATIONS);
	}

	public MultivariateProbabilityOptions(double absoluteTolerance,
			double relativeTolerance, int maxEvaluations, int replications) {
		this.absoluteTolerance = absoluteTolerance;
		this.relativeTolerance = relativeTolerance;
		this.maxEvaluations = maxEvaluations;
		this.replications = replications;
	}

	/** Returns the convergence tolerance for a finite probability estimate. */
	public double toleranceFor(double probability) {
		return Math.max(absoluteTolerance,
				relativeTolerance * Math.abs(probability));
	}

	/** Returns whether all tolerances and work limits are usable. */
	public boolean isValid() {
		return absoluteTolerance >= 0.0 && Double.isFinite(absoluteTolerance) &&
				relativeTolerance >= 0.0 && Double.isFinite(relativeTolerance) &&
				(absoluteTolerance > 0.0 || relativeTolerance > 0.0) &&
				replications >= 4 && maxEvaluations >= 2 * replications;
	}
}
