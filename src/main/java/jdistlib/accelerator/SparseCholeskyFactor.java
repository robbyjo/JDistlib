/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;

/** Immutable FP64 sparse Cholesky factor with reusable solves. */
public final class SparseCholeskyFactor {
	private final int dimension;
	private final double[] values;
	private final int[] columnIndices, rowStarts, permutation;
	private final double logDeterminant;

	SparseCholeskyFactor(int dimension, double[] values, int[] columnIndices,
			int[] rowStarts, int[] permutation) {
		if (dimension < 1 || values == null || columnIndices == null || rowStarts == null
				|| permutation == null || values.length != columnIndices.length
				|| rowStarts.length != dimension + 1 || permutation.length != dimension)
			throw new IllegalArgumentException("invalid sparse Cholesky storage");
		this.dimension = dimension; this.values = values.clone();
		this.columnIndices = columnIndices.clone(); this.rowStarts = rowStarts.clone();
		this.permutation = permutation.clone();
		double determinant = 0.0;
		for (int row = 0; row < dimension; row++) {
			double diagonal = diagonal(row);
			if (!(diagonal > 0.0)) throw new IllegalArgumentException("positive diagonal required");
			determinant += 2.0 * Math.log(diagonal);
		}
		logDeterminant = determinant;
	}

	public int dimension() { return dimension; }
	public int nonzeroCount() { return values.length; }
	public double logDeterminant() { return logDeterminant; }
	/** Returns the new-to-original symmetric permutation used by the factor. */
	public int[] permutation() { return permutation.clone(); }
	/** Returns the lower factor in permuted coordinates and one-based CSR storage. */
	public CsrMatrix lower() {
		return new CsrMatrix(dimension, dimension, values, columnIndices, rowStarts);
	}
	/** Solves {@code A*x=right} in original input coordinates. */
	public double[] solve(double[] right) {
		if (right == null || right.length != dimension)
			throw new IllegalArgumentException("right side length must equal factor dimension");
		double[] result = right.clone(); solveInPlace(result, 1); return result;
	}
	/** Solves a row-major dimension-by-columns right side in original coordinates. */
	public double[] solve(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid sparse Cholesky right side");
		double[] result = right.clone(); solveInPlace(result, columns); return result;
	}
	/** Replaces a row-major dimension-by-columns right side with its solution. */
	public void solveInPlace(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid sparse Cholesky right side");
		double[] work = new double[right.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(right, permutation[row] * columns, work, row * columns, columns);
		for (int row = 0; row < dimension; row++) {
			int start = rowStarts[row] - 1, end = rowStarts[row + 1] - 1;
			double diagonal = values[end - 1];
			for (int rhs = 0; rhs < columns; rhs++) {
				double value = work[row * columns + rhs];
				for (int offset = start; offset < end - 1; offset++) value -= values[offset]
						* work[(columnIndices[offset] - 1) * columns + rhs];
				work[row * columns + rhs] = value / diagonal;
			}
		}
		for (int row = dimension - 1; row >= 0; row--) {
			int end = rowStarts[row + 1] - 1; double diagonal = values[end - 1];
			for (int rhs = 0; rhs < columns; rhs++) work[row * columns + rhs] /= diagonal;
			for (int offset = rowStarts[row] - 1; offset < end - 1; offset++) {
				int column = columnIndices[offset] - 1;
				for (int rhs = 0; rhs < columns; rhs++) work[column * columns + rhs]
						-= values[offset] * work[row * columns + rhs];
			}
		}
		for (int row = 0; row < dimension; row++)
			System.arraycopy(work, row * columns, right, permutation[row] * columns, columns);
	}
	private double diagonal(int row) {
		int offset = rowStarts[row + 1] - 2;
		if (offset < rowStarts[row] - 1 || columnIndices[offset] != row + 1)
			throw new IllegalArgumentException("sparse Cholesky rows must end in their diagonal");
		return values[offset];
	}
}
