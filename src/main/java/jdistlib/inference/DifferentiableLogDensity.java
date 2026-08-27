/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** A log density capable of adding its gradient to caller-owned storage. */
public interface DifferentiableLogDensity extends LogDensity {
	/** Evaluates the log density and replaces {@code gradient} with its gradient. */
	double logDensityAndGradient(double[] state, double[] gradient);

	@Override default double logDensity(double[] state) {
		double[] gradient = new double[state.length];
		return logDensityAndGradient(state, gradient);
	}
}
