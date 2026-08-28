/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Thread-confined log-density wrapper reporting value and gradient work. */
public final class EvaluationCounter implements DifferentiableLogDensity {
	private final DifferentiableLogDensity target;
	private long logDensityEvaluations, gradientEvaluations;
	public EvaluationCounter(DifferentiableLogDensity target) {
		if (target == null) throw new IllegalArgumentException("target is required");
		this.target = target;
	}
	@Override public double logDensity(double[] state) {
		logDensityEvaluations++;
		return target.logDensity(state);
	}
	@Override public double logDensityAndGradient(double[] state, double[] gradient) {
		logDensityEvaluations++; gradientEvaluations++;
		return target.logDensityAndGradient(state, gradient);
	}
	public long logDensityEvaluations() { return logDensityEvaluations; }
	public long gradientEvaluations() { return gradientEvaluations; }
	public void reset() { logDensityEvaluations = 0L; gradientEvaluations = 0L; }
}
