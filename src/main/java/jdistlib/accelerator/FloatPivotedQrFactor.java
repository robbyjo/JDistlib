/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP32 column-pivoted Householder QR factorization. */
public final class FloatPivotedQrFactor {
	private final int rows, columns, reflectors; private final float[] qr, tau; private final int[] pivot;
	/** Creates a factor from packed Householder QR storage and a zero-based pivot. */
	public FloatPivotedQrFactor(int rows, int columns, float[] qr, float[] tau, int[] pivot) {
		int count = Math.min(rows, columns);
		if (rows < 1 || columns < 1 || qr == null || qr.length != rows * columns
				|| tau == null || tau.length != count || pivot == null || pivot.length != columns)
			throw new IllegalArgumentException("invalid FP32 pivoted QR dimensions");
		this.rows = rows; this.columns = columns; this.qr = qr.clone(); this.tau = tau.clone();
		this.pivot = pivot.clone(); reflectors = count;
	}
	public int rows() { return rows; }
	public int columns() { return columns; }
	public int[] pivot() { return pivot.clone(); }
	public float[] packed() { return qr.clone(); }
	public int rank() { float maximum = 0.0f; for (int i = 0; i < reflectors; i++) maximum = Math.max(maximum, Math.abs(qr[i * columns + i]));
		return rank(Math.max(rows, columns) * Math.ulp(1.0f) * maximum); }
	public int rank(float tolerance) { if (!(tolerance >= 0.0f) || Float.isInfinite(tolerance)) throw new IllegalArgumentException("rank tolerance must be finite and nonnegative");
		int value = 0; for (int i = 0; i < reflectors; i++) if (Math.abs(qr[i * columns + i]) > tolerance) value++; return value; }
	public float[] solveLeastSquares(float[] right) {
		if (rows < columns) throw new IllegalStateException("least-squares solve requires rows >= columns");
		if (right == null || right.length != rows) throw new IllegalArgumentException("right side length must equal matrix rows");
		if (rank() != columns) throw new IllegalStateException("least-squares matrix is rank deficient");
		float[] transformed = right.clone();
		for (int k = 0; k < reflectors; k++) { float product = transformed[k];
			for (int row = k + 1; row < rows; row++) product += qr[row * columns + k] * transformed[row];
			product *= tau[k]; transformed[k] -= product;
			for (int row = k + 1; row < rows; row++) transformed[row] -= qr[row * columns + k] * product; }
		float[] permuted = new float[columns];
		for (int row = columns - 1; row >= 0; row--) { float value = transformed[row];
			for (int column = row + 1; column < columns; column++) value -= qr[row * columns + column] * permuted[column];
			permuted[row] = value / qr[row * columns + row]; }
		float[] result = new float[columns]; for (int column = 0; column < columns; column++) result[pivot[column]] = permuted[column]; return result;
	}
}
