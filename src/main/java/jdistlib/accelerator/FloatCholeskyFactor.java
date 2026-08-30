/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP32 lower Cholesky factor with reusable SPD solves. */
public final class FloatCholeskyFactor {
	private final int dimension;
	private final float[] lower;
	private final float logDeterminant;
	/** Creates a factor from a row-major lower-triangular matrix. */
	public FloatCholeskyFactor(int dimension, float[] lower) {
		if (dimension < 1 || lower == null || lower.length != dimension * dimension)
			throw new IllegalArgumentException("invalid FP32 Cholesky factor dimensions");
		this.dimension = dimension; this.lower = lower.clone(); float value = 0.0f;
		for (int i = 0; i < dimension; i++) value += 2.0f * (float) Math.log(this.lower[i * dimension + i]);
		logDeterminant = value;
	}
	public int dimension() { return dimension; }
	public float[] lower() { return lower.clone(); }
	public float logDeterminant() { return logDeterminant; }
	public float[] solve(float[] right) {
		if (right == null || right.length != dimension)
			throw new IllegalArgumentException("right side length must equal factor dimension");
		float[] result = right.clone();
		for (int row = 0; row < dimension; row++) { float value = result[row];
			for (int column = 0; column < row; column++) value -= lower[row * dimension + column] * result[column];
			result[row] = value / lower[row * dimension + row]; }
		for (int row = dimension - 1; row >= 0; row--) { float value = result[row];
			for (int column = row + 1; column < dimension; column++) value -= lower[column * dimension + row] * result[column];
			result[row] = value / lower[row * dimension + row]; }
		return result;
	}
	/** Solves one or more row-major right-hand-side columns. */
	public float[] solve(float[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("right side must contain dimension by columns values");
		float[] result = right.clone();
		for (int row = 0; row < dimension; row++) for (int rhs = 0; rhs < columns; rhs++) {
			float value = result[row * columns + rhs];
			for (int column = 0; column < row; column++)
				value -= lower[row * dimension + column] * result[column * columns + rhs];
			result[row * columns + rhs] = value / lower[row * dimension + row];
		}
		for (int row = dimension - 1; row >= 0; row--) for (int rhs = 0; rhs < columns; rhs++) {
			float value = result[row * columns + rhs];
			for (int column = row + 1; column < dimension; column++)
				value -= lower[column * dimension + row] * result[column * columns + rhs];
			result[row * columns + rhs] = value / lower[row * dimension + row];
		}
		return result;
	}
}
