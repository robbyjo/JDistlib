/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.matrix;

/** Immutable FP32 CSR matrix with one-based Stan-compatible indices. */
public final class FloatCsrMatrix {
	private final int rows, columns;
	private final float[] values;
	private final int[] columnIndices, rowStarts;

	public FloatCsrMatrix(int rows, int columns, float[] values,
			int[] columnIndices, int[] rowStarts) {
		if (rows < 0 || columns < 0 || values == null || columnIndices == null
				|| rowStarts == null || values.length != columnIndices.length
				|| rowStarts.length != rows + 1)
			throw new IllegalArgumentException("invalid FP32 CSR dimensions or storage lengths");
		if (rowStarts[0] != 1 || rowStarts[rows] != values.length + 1)
			throw new IllegalArgumentException("CSR row starts must be one-based and terminate at nnz+1");
		for (int i = 0; i < rows; i++) if (rowStarts[i] > rowStarts[i + 1])
			throw new IllegalArgumentException("CSR row starts must be nondecreasing");
		for (int index : columnIndices) if (index < 1 || index > columns)
			throw new IllegalArgumentException("CSR column index outside 1..columns");
		this.rows = rows; this.columns = columns; this.values = values.clone();
		this.columnIndices = columnIndices.clone(); this.rowStarts = rowStarts.clone();
	}

	public int rows() { return rows; }
	public int columns() { return columns; }
	public int nonzeroCount() { return values.length; }
	public float[] values() { return values.clone(); }
	public int[] columnIndices() { return columnIndices.clone(); }
	public int[] rowStarts() { return rowStarts.clone(); }
}
