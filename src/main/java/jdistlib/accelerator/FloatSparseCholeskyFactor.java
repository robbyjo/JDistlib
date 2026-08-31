/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.FloatCsrMatrix;

/** Immutable FP32 sparse Cholesky factor with reusable solves. */
public final class FloatSparseCholeskyFactor {
	private final int dimension;
	private final float[] values;
	private final int[] columnIndices, rowStarts, permutation;
	private final float logDeterminant;

	FloatSparseCholeskyFactor(int dimension, float[] values, int[] columnIndices,
			int[] rowStarts, int[] permutation) {
		if (dimension < 1 || values == null || columnIndices == null || rowStarts == null
				|| permutation == null || values.length != columnIndices.length
				|| rowStarts.length != dimension + 1 || permutation.length != dimension)
			throw new IllegalArgumentException("invalid FP32 sparse Cholesky storage");
		this.dimension = dimension; this.values = values.clone();
		this.columnIndices = columnIndices.clone(); this.rowStarts = rowStarts.clone();
		this.permutation = permutation.clone(); float determinant = 0.0f;
		for (int row = 0; row < dimension; row++) {
			float diagonal = diagonal(row);
			if (!(diagonal > 0.0f)) throw new IllegalArgumentException("positive diagonal required");
			determinant += 2.0f * (float) Math.log(diagonal);
		}
		logDeterminant = determinant;
	}

	public int dimension() { return dimension; }
	public int nonzeroCount() { return values.length; }
	public float logDeterminant() { return logDeterminant; }
	public int[] permutation() { return permutation.clone(); }
	public FloatCsrMatrix lower() {
		return new FloatCsrMatrix(dimension, dimension, values, columnIndices, rowStarts);
	}
	public float[] solve(float[] right) {
		if (right == null || right.length != dimension)
			throw new IllegalArgumentException("right side length must equal factor dimension");
		float[] result = right.clone(); solveInPlace(result, 1); return result;
	}
	public float[] solve(float[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid FP32 sparse Cholesky right side");
		float[] result = right.clone(); solveInPlace(result, columns); return result;
	}
	public void solveInPlace(float[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid FP32 sparse Cholesky right side");
		float[] work = new float[right.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(right, permutation[row] * columns, work, row * columns, columns);
		for (int row = 0; row < dimension; row++) {
			int start = rowStarts[row] - 1, end = rowStarts[row + 1] - 1;
			float diagonal = values[end - 1];
			for (int rhs = 0; rhs < columns; rhs++) {
				float value = work[row * columns + rhs];
				for (int offset = start; offset < end - 1; offset++) value -= values[offset]
						* work[(columnIndices[offset] - 1) * columns + rhs];
				work[row * columns + rhs] = value / diagonal;
			}
		}
		for (int row = dimension - 1; row >= 0; row--) {
			int end = rowStarts[row + 1] - 1; float diagonal = values[end - 1];
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
	private float diagonal(int row) {
		int offset = rowStarts[row + 1] - 2;
		if (offset < rowStarts[row] - 1 || columnIndices[offset] != row + 1)
			throw new IllegalArgumentException("sparse Cholesky rows must end in their diagonal");
		return values[offset];
	}
}
