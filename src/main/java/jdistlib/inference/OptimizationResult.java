/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable numerical-optimization result. */
public final class OptimizationResult {
	private final double[] point;
	private final double objective;
	private final int iterations, evaluations;
	private final boolean converged;
	OptimizationResult(double[] point, double objective, int iterations,
			int evaluations, boolean converged) {
		this.point = point.clone(); this.objective = objective;
		this.iterations = iterations; this.evaluations = evaluations;
		this.converged = converged;
	}
	public double[] point() { return point.clone(); }
	public double objective() { return objective; }
	public int iterations() { return iterations; }
	public int evaluations() { return evaluations; }
	public boolean converged() { return converged; }
}
