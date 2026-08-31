/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP32 partial-pivoted LU factorization of a square matrix. */
public final class FloatLuFactor {
	private final int dimension;
	private final float[] packed;
	private final int[] permutation;
	private final int determinantSign;
	public FloatLuFactor(int dimension, float[] packed, int[] permutation, int determinantSign) {
		if (dimension < 1 || packed == null || packed.length != dimension * dimension
				|| permutation == null || permutation.length != dimension
				|| (determinantSign != -1 && determinantSign != 1))
			throw new IllegalArgumentException("invalid FP32 LU factorization");
		boolean[] seen = new boolean[dimension];
		for (int value : permutation) {
			if (value < 0 || value >= dimension || seen[value])
				throw new IllegalArgumentException("FP32 LU permutation must contain each row once");
			seen[value] = true;
		}
		this.dimension = dimension; this.packed = packed.clone();
		this.permutation = permutation.clone(); this.determinantSign = determinantSign;
	}
	public int dimension() { return dimension; }
	public float[] packed() { return packed.clone(); }
	public int[] permutation() { return permutation.clone(); }
	public int determinantSign() { int sign = determinantSign; for (int i = 0; i < dimension; i++)
		if (packed[i * dimension + i] < 0.0f) sign = -sign; return sign; }
	public float logAbsDeterminant() { float value = 0.0f; for (int i = 0; i < dimension; i++)
		value += (float) Math.log(Math.abs(packed[i * dimension + i])); return value; }
	public float[] solve(float[] right) { return solve(right, 1); }
	public float[] solve(float[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid FP32 LU right side");
		float[] result = new float[right.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(right, permutation[row] * columns, result, row * columns, columns);
		for (int row = 0; row < dimension; row++) for (int column = 0; column < columns; column++) {
			float value = result[row * columns + column];
			for (int k = 0; k < row; k++) value -= packed[row * dimension + k] * result[k * columns + column];
			result[row * columns + column] = value;
		}
		for (int row = dimension - 1; row >= 0; row--) for (int column = 0; column < columns; column++) {
			float value = result[row * columns + column];
			for (int k = row + 1; k < dimension; k++) value -= packed[row * dimension + k] * result[k * columns + column];
			result[row * columns + column] = value / packed[row * dimension + row];
		}
		return result;
	}
}
