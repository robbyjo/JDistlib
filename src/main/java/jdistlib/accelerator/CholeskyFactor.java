/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable lower Cholesky factor with reusable SPD solves. */
public final class CholeskyFactor {
	private final int dimension;
	private final double[] lower;
	private final double logDeterminant;

	/** Creates a factor from a row-major lower-triangular matrix. */
	public CholeskyFactor(int dimension, double[] lower) {
		if (dimension < 1 || lower == null || lower.length != dimension * dimension)
			throw new IllegalArgumentException("invalid Cholesky factor dimensions");
		this.dimension = dimension;
		this.lower = lower.clone();
		double value = 0.0;
		for (int i = 0; i < dimension; i++) value += 2.0 * Math.log(this.lower[i * dimension + i]);
		logDeterminant = value;
	}

	/** Returns the matrix dimension. */
	public int dimension() { return dimension; }
	/** Returns a row-major copy of the lower-triangular factor. */
	public double[] lower() { return lower.clone(); }
	/** Returns the log determinant of the original SPD matrix. */
	public double logDeterminant() { return logDeterminant; }

	/** Solves {@code A*x=right}. */
	public double[] solve(double[] right) {
		if (right == null || right.length != dimension)
			throw new IllegalArgumentException("right side length must equal factor dimension");
		double[] result = right.clone();
		for (int row = 0; row < dimension; row++) {
			double value = result[row];
			for (int column = 0; column < row; column++)
				value -= lower[row * dimension + column] * result[column];
			result[row] = value / lower[row * dimension + row];
		}
		for (int row = dimension - 1; row >= 0; row--) {
			double value = result[row];
			for (int column = row + 1; column < dimension; column++)
				value -= lower[column * dimension + row] * result[column];
			result[row] = value / lower[row * dimension + row];
		}
		return result;
	}

	/** Solves {@code A*X=right} for a row-major matrix with the given column count. */
	public double[] solve(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("right side must have factor dimension rows");
		double[] result = right.clone();
		for (int column = 0; column < columns; column++) {
			for (int row = 0; row < dimension; row++) {
				double value = result[row * columns + column];
				for (int previous = 0; previous < row; previous++)
					value -= lower[row * dimension + previous]
							* result[previous * columns + column];
				result[row * columns + column] = value / lower[row * dimension + row];
			}
			for (int row = dimension - 1; row >= 0; row--) {
				double value = result[row * columns + column];
				for (int later = row + 1; later < dimension; later++)
					value -= lower[later * dimension + row]
							* result[later * columns + column];
				result[row * columns + column] = value / lower[row * dimension + row];
			}
		}
		return result;
	}
}
