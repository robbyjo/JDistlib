/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Immutable diagnostics for an approximate scalar calculation. */
public final class NumericalEstimate {
	private final double value;
	private final double absoluteError;
	private final boolean converged;
	private final int evaluations;
	private final String strategy;
	private final String warning;

	public NumericalEstimate(double value, double absoluteError, boolean converged,
			int evaluations, String strategy, String warning) {
		this.value = value;
		this.absoluteError = absoluteError;
		this.converged = converged;
		this.evaluations = evaluations;
		this.strategy = strategy == null ? "unspecified" : strategy;
		this.warning = warning == null ? "" : warning;
	}

	public double getValue() { return value; }
	public double getAbsoluteError() { return absoluteError; }
	public boolean isConverged() { return converged; }
	public int getEvaluations() { return evaluations; }
	public String getStrategy() { return strategy; }
	public String getWarning() { return warning; }
	public boolean hasWarning() { return !warning.isEmpty(); }

	/** Returns the value only after enforcing the checked-result contract. */
	public double valueOrThrow() {
		if (!converged) throw new IllegalStateException("calculation did not converge: " + warning);
		return value;
	}
}
