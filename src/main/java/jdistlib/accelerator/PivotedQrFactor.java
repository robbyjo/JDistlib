/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable column-pivoted Householder QR factorization. */
public final class PivotedQrFactor {
	private final int rows, columns, reflectors;
	private final double[] qr, tau;
	private final int[] pivot;

	/** Creates a factor from packed Householder QR storage and a zero-based pivot. */
	public PivotedQrFactor(int rows, int columns, double[] qr, double[] tau, int[] pivot) {
		int count = Math.min(rows, columns);
		if (rows < 1 || columns < 1 || qr == null || qr.length != rows * columns
				|| tau == null || tau.length != count || pivot == null || pivot.length != columns)
			throw new IllegalArgumentException("invalid pivoted QR dimensions");
		this.rows = rows; this.columns = columns; this.qr = qr.clone();
		this.tau = tau.clone(); this.pivot = pivot.clone(); reflectors = count;
	}

	/** Returns the number of rows in the original matrix. */
	public int rows() { return rows; }
	/** Returns the number of columns in the original matrix. */
	public int columns() { return columns; }
	/** Returns the permutation where factor column {@code j} came from original column {@code pivot[j]}. */
	public int[] pivot() { return pivot.clone(); }
	/** Returns a row-major copy containing R and the packed Householder vectors. */
	public double[] packed() { return qr.clone(); }

	/** Estimates numerical rank using {@code max(rows,columns)*ulp(1)*max(abs(diag(R)))}. */
	public int rank() {
		double maximum = 0.0;
		for (int i = 0; i < reflectors; i++) maximum = Math.max(maximum, Math.abs(qr[i * columns + i]));
		return rank(Math.max(rows, columns) * Math.ulp(1.0) * maximum);
	}

	/** Returns the number of diagonal entries of R larger than an absolute tolerance. */
	public int rank(double tolerance) {
		if (!(tolerance >= 0.0) || Double.isInfinite(tolerance))
			throw new IllegalArgumentException("rank tolerance must be finite and nonnegative");
		int value = 0;
		for (int i = 0; i < reflectors; i++) if (Math.abs(qr[i * columns + i]) > tolerance) value++;
		return value;
	}

	/** Solves a full-column-rank least-squares problem and returns coefficients in original order. */
	public double[] solveLeastSquares(double[] right) {
		if (rows < columns) throw new IllegalStateException("least-squares solve requires rows >= columns");
		if (right == null || right.length != rows)
			throw new IllegalArgumentException("right side length must equal matrix rows");
		if (rank() != columns) throw new IllegalStateException("least-squares matrix is rank deficient");
		double[] transformed = right.clone();
		for (int k = 0; k < reflectors; k++) {
			double product = transformed[k];
			for (int row = k + 1; row < rows; row++)
				product += qr[row * columns + k] * transformed[row];
			product *= tau[k]; transformed[k] -= product;
			for (int row = k + 1; row < rows; row++)
				transformed[row] -= qr[row * columns + k] * product;
		}
		double[] permuted = new double[columns];
		for (int row = columns - 1; row >= 0; row--) {
			double value = transformed[row];
			for (int column = row + 1; column < columns; column++)
				value -= qr[row * columns + column] * permuted[column];
			permuted[row] = value / qr[row * columns + row];
		}
		double[] result = new double[columns];
		for (int column = 0; column < columns; column++) result[pivot[column]] = permuted[column];
		return result;
	}
}
