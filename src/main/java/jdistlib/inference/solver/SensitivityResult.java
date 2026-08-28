/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Values and first-order parameter sensitivities returned by a numerical solver. */
public final class SensitivityResult {
	private final double[][] values;
	private final double[][][] sensitivities;

	SensitivityResult(double[][] values, double[][][] sensitivities) {
		this.values = copy(values); this.sensitivities = copy(sensitivities);
	}

	/** Output values indexed by output point then state component. */
	public double[][] values() { return copy(values); }

	/** Derivatives indexed by output point, state component, then parameter. */
	public double[][][] sensitivities() { return copy(sensitivities); }

	private static double[][] copy(double[][] source) {
		double[][] result = new double[source.length][];
		for (int i = 0; i < source.length; i++) result[i] = source[i].clone();
		return result;
	}
	private static double[][][] copy(double[][][] source) {
		double[][][] result = new double[source.length][][];
		for (int i = 0; i < source.length; i++) result[i] = copy(source[i]);
		return result;
	}
}
