/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Quasi-Newton Gaussian initialization result. */
public final class PathfinderResult {
	private final double[] initialState, mode, scale;
	private final OptimizationResult optimization;
	PathfinderResult(double[] initialState, double[] mode, double[] scale,
			OptimizationResult optimization) {
		this.initialState = initialState.clone(); this.mode = mode.clone();
		this.scale = scale.clone(); this.optimization = optimization;
	}
	public double[] initialState() { return initialState.clone(); }
	public double[] mode() { return mode.clone(); }
	public double[] marginalScale() { return scale.clone(); }
	public OptimizationResult optimization() { return optimization; }
}
