/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP32 thin singular-value decomposition {@code A = U*S*Vt}. */
public final class FloatSingularValueDecomposition {
	private final int rows, columns, components;
	private final float[] singularValues, left, rightTransposed;

	/** Creates a thin SVD with descending singular values. */
	public FloatSingularValueDecomposition(int rows, int columns, float[] singularValues,
			float[] left, float[] rightTransposed) {
		int count = Math.min(rows, columns);
		if (rows < 1 || columns < 1 || singularValues == null || singularValues.length != count
				|| left == null || left.length != rows * count || rightTransposed == null
				|| rightTransposed.length != count * columns)
			throw new IllegalArgumentException("invalid FP32 thin SVD dimensions");
		this.rows = rows; this.columns = columns; components = count;
		this.singularValues = singularValues.clone(); this.left = left.clone();
		this.rightTransposed = rightTransposed.clone();
	}

	public int rows() { return rows; }
	public int columns() { return columns; }
	public int components() { return components; }
	/** Returns singular values in descending order. */
	public float[] singularValues() { return singularValues.clone(); }
	/** Returns row-major thin U with shape rows by components. */
	public float[] leftSingularVectors() { return left.clone(); }
	/** Returns row-major thin V-transpose with shape components by columns. */
	public float[] rightSingularVectorsTransposed() { return rightTransposed.clone(); }
	/** Returns numerical rank using the standard dimension-scaled machine threshold. */
	public int rank() {
		float threshold = Math.max(rows, columns) * Math.ulp(1.0f) * singularValues[0];
		int rank = 0; for (float value : singularValues) if (value > threshold) rank++; return rank;
	}
}
