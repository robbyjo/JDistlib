/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Numerical controls for mixed continuous/discrete copula likelihoods. */
public final class CopulaMeasureOptions {
	private final double derivativeStep;
	private final double minimumStep;
	private final int maxCdfEvaluations;
	private final double negativeTolerance;

	public CopulaMeasureOptions() { this(1e-4, 1e-7, 1_048_576, 1e-10); }

	private CopulaMeasureOptions(double derivativeStep, double minimumStep,
			int maxCdfEvaluations, double negativeTolerance) {
		if (!(derivativeStep > 0.0 && derivativeStep < 0.25)
				|| !(minimumStep > 0.0 && minimumStep <= derivativeStep)
				|| maxCdfEvaluations < 2 || !(negativeTolerance >= 0.0)
				|| !Double.isFinite(derivativeStep) || !Double.isFinite(minimumStep)
				|| !Double.isFinite(negativeTolerance)) {
			throw new IllegalArgumentException("invalid mixed-measure options");
		}
		this.derivativeStep = derivativeStep;
		this.minimumStep = minimumStep;
		this.maxCdfEvaluations = maxCdfEvaluations;
		this.negativeTolerance = negativeTolerance;
	}

	public double getDerivativeStep() { return derivativeStep; }
	public double getMinimumStep() { return minimumStep; }
	public int getMaxCdfEvaluations() { return maxCdfEvaluations; }
	public double getNegativeTolerance() { return negativeTolerance; }

	public CopulaMeasureOptions withDerivativeSteps(double step, double minimum) {
		return new CopulaMeasureOptions(step, minimum, maxCdfEvaluations, negativeTolerance);
	}

	public CopulaMeasureOptions withMaxCdfEvaluations(int maximum) {
		return new CopulaMeasureOptions(derivativeStep, minimumStep, maximum,
				negativeTolerance);
	}

	public CopulaMeasureOptions withNegativeTolerance(double tolerance) {
		return new CopulaMeasureOptions(derivativeStep, minimumStep,
				maxCdfEvaluations, tolerance);
	}
}
