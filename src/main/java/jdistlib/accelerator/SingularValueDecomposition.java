/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP64 thin singular-value decomposition {@code A = U*S*Vt}. */
public final class SingularValueDecomposition {
	private final int rows, columns, components;
	private final double[] singularValues, left, rightTransposed;

	/** Creates a thin SVD with descending singular values. */
	public SingularValueDecomposition(int rows, int columns, double[] singularValues,
			double[] left, double[] rightTransposed) {
		int count = Math.min(rows, columns);
		if (rows < 1 || columns < 1 || singularValues == null || singularValues.length != count
				|| left == null || left.length != rows * count || rightTransposed == null
				|| rightTransposed.length != count * columns)
			throw new IllegalArgumentException("invalid thin SVD dimensions");
		this.rows = rows; this.columns = columns; components = count;
		this.singularValues = singularValues.clone(); this.left = left.clone();
		this.rightTransposed = rightTransposed.clone();
	}

	public int rows() { return rows; }
	public int columns() { return columns; }
	public int components() { return components; }
	/** Returns singular values in descending order. */
	public double[] singularValues() { return singularValues.clone(); }
	/** Returns row-major thin U with shape rows by components. */
	public double[] leftSingularVectors() { return left.clone(); }
	/** Returns row-major thin V-transpose with shape components by columns. */
	public double[] rightSingularVectorsTransposed() { return rightTransposed.clone(); }
	/** Returns numerical rank using the standard dimension-scaled machine threshold. */
	public int rank() {
		double threshold = Math.max(rows, columns) * Math.ulp(1.0) * singularValues[0];
		int rank = 0; for (double value : singularValues) if (value > threshold) rank++; return rank;
	}
}
