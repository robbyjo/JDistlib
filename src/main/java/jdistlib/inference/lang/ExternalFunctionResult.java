/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

/** Immutable external-function values and derivatives with respect to flattened arguments. */
public final class ExternalFunctionResult {
	private final double[] values;
	private final int[] shape;
	private final double[][] jacobian;

	/**
	 * @param values flattened return values
	 * @param shape return dimensions, empty for a scalar
	 * @param jacobian rows by return value and columns by all argument elements in order
	 */
	public ExternalFunctionResult(double[] values, int[] shape, double[][] jacobian) {
		if (values == null || shape == null || jacobian == null || jacobian.length != values.length)
			throw new IllegalArgumentException("external values, shape, and Jacobian rows are required");
		long count = 1;
		for (int extent : shape) { if (extent < 0) throw new IllegalArgumentException("negative result extent"); count *= extent; }
		if (count != values.length) throw new IllegalArgumentException("external result shape does not match values");
		int columns = jacobian.length == 0 ? 0 : jacobian[0].length;
		this.jacobian = new double[jacobian.length][];
		for (int row = 0; row < jacobian.length; row++) {
			if (jacobian[row] == null || jacobian[row].length != columns)
				throw new IllegalArgumentException("external Jacobian must be rectangular");
			this.jacobian[row] = jacobian[row].clone();
		}
		this.values = values.clone(); this.shape = shape.clone();
	}

	public static ExternalFunctionResult scalar(double value, double... partials) {
		return new ExternalFunctionResult(new double[] {value}, new int[0], new double[][] {partials});
	}
	public double[] values() { return values.clone(); }
	public int[] shape() { return shape.clone(); }
	public double[][] jacobian() {
		double[][] result = new double[jacobian.length][];
		for (int i = 0; i < result.length; i++) result[i] = jacobian[i].clone(); return result;
	}
}
