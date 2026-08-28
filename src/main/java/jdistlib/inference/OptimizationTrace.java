/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Accepted L-BFGS path used by Pathfinder approximation selection. */
public final class OptimizationTrace {
	private final double[][] points; private final double[] objectives; private final OptimizationResult result;
	private final double[][][] inverseHessians;
	OptimizationTrace(double[][] points, double[] objectives, double[][][] inverseHessians, OptimizationResult result) {
		this.points = copy(points); this.objectives = objectives.clone(); this.inverseHessians = copy(inverseHessians); this.result = result; }
	public double[][] points() { return copy(points); } public double[] objectives() { return objectives.clone(); }
	public double[][][] inverseHessians() { return copy(inverseHessians); }
	public OptimizationResult result() { return result; }
	private static double[][] copy(double[][] values) { double[][] result = new double[values.length][]; for (int i = 0; i < values.length; i++) result[i] = values[i].clone(); return result; }
	private static double[][][] copy(double[][][] values) { double[][][] result = new double[values.length][][]; for (int i = 0; i < values.length; i++) result[i] = copy(values[i]); return result; }
}
