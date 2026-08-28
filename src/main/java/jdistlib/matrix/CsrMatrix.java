/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.matrix;

import java.util.Arrays;

/** Immutable one-based compressed-row sparse matrix compatible with Stan CSR arrays. */
public final class CsrMatrix {
	private final int rows, columns;
	private final double[] values;
	private final int[] columnIndices, rowStarts;

	public CsrMatrix(int rows, int columns, double[] values,
			int[] columnIndices, int[] rowStarts) {
		if (rows < 0 || columns < 0 || values == null || columnIndices == null || rowStarts == null
				|| values.length != columnIndices.length || rowStarts.length != rows+1)
			throw new IllegalArgumentException("invalid CSR dimensions or storage lengths");
		if (rowStarts[0] != 1 || rowStarts[rows] != values.length+1)
			throw new IllegalArgumentException("CSR row starts must be one-based and terminate at nnz+1");
		for (int i = 0; i < rows; i++) if (rowStarts[i] > rowStarts[i+1])
			throw new IllegalArgumentException("CSR row starts must be nondecreasing");
		for (int index : columnIndices) if (index < 1 || index > columns)
			throw new IllegalArgumentException("CSR column index outside 1..columns");
		this.rows = rows; this.columns = columns; this.values = values.clone();
		this.columnIndices = columnIndices.clone(); this.rowStarts = rowStarts.clone();
	}

	public int rows() { return rows; }
	public int columns() { return columns; }
	public int nonzeroCount() { return values.length; }
	public double[] values() { return values.clone(); }
	public int[] columnIndices() { return columnIndices.clone(); }
	public int[] rowStarts() { return rowStarts.clone(); }

	/** Multiplies this sparse matrix by a dense vector. */
	public double[] multiply(double[] vector) {
		if (vector == null || vector.length != columns)
			throw new IllegalArgumentException("vector length must equal sparse column count");
		double[] result = new double[rows];
		for (int row = 0; row < rows; row++) for (int offset = rowStarts[row]-1; offset < rowStarts[row+1]-1; offset++)
			result[row] += values[offset]*vector[columnIndices[offset]-1];
		return result;
	}

	/** Returns a row-major dense matrix. */
	public double[] toDense() {
		double[] result = new double[rows*columns];
		for (int row = 0; row < rows; row++) for (int offset = rowStarts[row]-1; offset < rowStarts[row+1]-1; offset++)
			result[row*columns+columnIndices[offset]-1] += values[offset];
		return result;
	}

	/** Converts a row-major dense matrix, omitting entries with absolute value at most {@code tolerance}. */
	public static CsrMatrix fromDense(int rows, int columns, double[] dense, double tolerance) {
		if (rows < 0 || columns < 0 || dense == null || dense.length != rows*columns || tolerance < 0)
			throw new IllegalArgumentException("invalid dense matrix or tolerance");
		int count = 0; for (double value : dense) if (Math.abs(value) > tolerance) count++;
		double[] values = new double[count]; int[] indices = new int[count], starts = new int[rows+1];
		int offset = 0; starts[0] = 1;
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				double value = dense[row*columns+column];
				if (Math.abs(value) > tolerance) { values[offset] = value; indices[offset++] = column+1; }
			}
			starts[row+1] = offset+1;
		}
		return new CsrMatrix(rows, columns, values, indices, starts);
	}

	@Override public String toString() {
		return "CsrMatrix("+rows+"x"+columns+", nnz="+values.length+", u="+Arrays.toString(rowStarts)+")";
	}
}
