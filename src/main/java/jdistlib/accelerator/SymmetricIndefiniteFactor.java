/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable pivoted {@code P*A*P' = L*D*L'} factorization with 1x1/2x2 D blocks. */
public final class SymmetricIndefiniteFactor {
	private final int dimension;
	private final double[] lower, diagonal;
	private final int[] permutation, blockSizes;
	public SymmetricIndefiniteFactor(int dimension, double[] lower, double[] diagonal,
			int[] permutation, int[] blockSizes) {
		if (dimension < 1 || lower == null || lower.length != dimension * dimension
				|| diagonal == null || diagonal.length != dimension * dimension
				|| permutation == null || permutation.length != dimension
				|| blockSizes == null || blockSizes.length != dimension)
			throw new IllegalArgumentException("invalid symmetric-indefinite factorization");
		boolean[] seen = new boolean[dimension];
		for (int value : permutation) {
			if (value < 0 || value >= dimension || seen[value])
				throw new IllegalArgumentException("LDL permutation must contain each row once");
			seen[value] = true;
		}
		for (int i = 0; i < dimension;) {
			if (blockSizes[i] != 1 && blockSizes[i] != 2)
				throw new IllegalArgumentException("LDL block starts must have size one or two");
			if (blockSizes[i] == 2 && (i + 1 >= dimension || blockSizes[i + 1] != 0))
				throw new IllegalArgumentException("invalid LDL 2x2 block marker");
			i += blockSizes[i];
		}
		this.dimension = dimension; this.lower = lower.clone(); this.diagonal = diagonal.clone();
		this.permutation = permutation.clone(); this.blockSizes = blockSizes.clone();
	}
	public int dimension() { return dimension; }
	public double[] lower() { return lower.clone(); }
	public double[] diagonalBlocks() { return diagonal.clone(); }
	public int[] permutation() { return permutation.clone(); }
	public int[] blockSizes() { return blockSizes.clone(); }
	public int determinantSign() {
		int sign = 1;
		for (int i = 0; i < dimension;) {
			double determinant;
			if (blockSizes[i] == 2) determinant = diagonal[i * dimension + i]
					* diagonal[(i + 1) * dimension + i + 1] - diagonal[i * dimension + i + 1]
					* diagonal[(i + 1) * dimension + i];
			else determinant = diagonal[i * dimension + i];
			if (determinant < 0.0) sign = -sign; i += blockSizes[i];
		}
		return sign;
	}
	public double logAbsDeterminant() {
		double value = 0.0;
		for (int i = 0; i < dimension;) {
			double determinant = blockSizes[i] == 2
					? diagonal[i * dimension + i] * diagonal[(i + 1) * dimension + i + 1]
						- diagonal[i * dimension + i + 1] * diagonal[(i + 1) * dimension + i]
					: diagonal[i * dimension + i];
			value += Math.log(Math.abs(determinant)); i += blockSizes[i];
		}
		return value;
	}
	public double[] solve(double[] right) { return solve(right, 1); }
	public double[] solve(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid symmetric-indefinite right side");
		double[] work = new double[right.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(right, permutation[row] * columns, work, row * columns, columns);
		for (int row = 0; row < dimension; row++) for (int column = 0; column < columns; column++) {
			double value = work[row * columns + column];
			for (int k = 0; k < row; k++) value -= lower[row * dimension + k] * work[k * columns + column];
			work[row * columns + column] = value;
		}
		for (int i = 0; i < dimension;) {
			if (blockSizes[i] == 2) {
				double a = diagonal[i * dimension + i], b = diagonal[i * dimension + i + 1];
				double c = diagonal[(i + 1) * dimension + i + 1], determinant = a * c - b * b;
				for (int column = 0; column < columns; column++) {
					double first = work[i * columns + column], second = work[(i + 1) * columns + column];
					work[i * columns + column] = (c * first - b * second) / determinant;
					work[(i + 1) * columns + column] = (a * second - b * first) / determinant;
				}
				i += 2;
			} else {
				for (int column = 0; column < columns; column++)
					work[i * columns + column] /= diagonal[i * dimension + i];
				i++;
			}
		}
		for (int row = dimension - 1; row >= 0; row--) for (int column = 0; column < columns; column++) {
			double value = work[row * columns + column];
			for (int k = row + 1; k < dimension; k++) value -= lower[k * dimension + row] * work[k * columns + column];
			work[row * columns + column] = value;
		}
		double[] result = new double[work.length];
		for (int row = 0; row < dimension; row++)
			System.arraycopy(work, row * columns, result, permutation[row] * columns, columns);
		return result;
	}
}
