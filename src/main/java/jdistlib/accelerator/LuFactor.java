/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP64 partial-pivoted LU factorization of a square matrix. */
public final class LuFactor {
	private final int dimension;
	private final double[] packed;
	private final int[] permutation;
	private final int determinantSign;

	/** Creates a factor where row {@code i} came from original row {@code permutation[i]}. */
	public LuFactor(int dimension, double[] packed, int[] permutation, int determinantSign) {
		if (dimension < 1 || packed == null || packed.length != dimension * dimension
				|| permutation == null || permutation.length != dimension
				|| (determinantSign != -1 && determinantSign != 1))
			throw new IllegalArgumentException("invalid LU factorization");
		boolean[] seen = new boolean[dimension];
		for (int value : permutation) {
			if (value < 0 || value >= dimension || seen[value])
				throw new IllegalArgumentException("LU permutation must contain each row once");
			seen[value] = true;
		}
		this.dimension = dimension;
		this.packed = packed.clone();
		this.permutation = permutation.clone();
		this.determinantSign = determinantSign;
	}

	public int dimension() { return dimension; }
	/** Returns packed unit-lower and upper factors in row-major storage. */
	public double[] packed() { return packed.clone(); }
	/** Returns the new-to-original row permutation. */
	public int[] permutation() { return permutation.clone(); }
	public int determinantSign() {
		int sign = determinantSign;
		for (int i = 0; i < dimension; i++) if (packed[i * dimension + i] < 0.0) sign = -sign;
		return sign;
	}
	public double logAbsDeterminant() {
		double value = 0.0;
		for (int i = 0; i < dimension; i++) value += Math.log(Math.abs(packed[i * dimension + i]));
		return value;
	}
	public double[] solve(double[] right) { return solve(right, 1); }
	/** Solves {@code A*X=right}; right sides use row-major dimension-by-columns storage. */
	public double[] solve(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid LU right side");
		double[] result = new double[right.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(right, permutation[row] * columns, result, row * columns, columns);
		for (int row = 0; row < dimension; row++) for (int column = 0; column < columns; column++) {
			double value = result[row * columns + column];
			for (int k = 0; k < row; k++) value -= packed[row * dimension + k] * result[k * columns + column];
			result[row * columns + column] = value;
		}
		for (int row = dimension - 1; row >= 0; row--) for (int column = 0; column < columns; column++) {
			double value = result[row * columns + column];
			for (int k = row + 1; k < dimension; k++) value -= packed[row * dimension + k] * result[k * columns + column];
			result[row * columns + column] = value / packed[row * dimension + row];
		}
		return result;
	}
}
