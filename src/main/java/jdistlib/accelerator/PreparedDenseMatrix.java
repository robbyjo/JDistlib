/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP64 dense matrix handle; providers may retain storage on device. */
public interface PreparedDenseMatrix extends AutoCloseable {
	int rows();
	int columns();
	/** Performs {@code C := alpha*op(A)*B + beta*C} with row-major B and C. */
	void multiply(MatrixTranspose transpose, double alpha, double[] right, int rightColumns,
			double beta, double[] result);
	default void multiplyBatched(MatrixTranspose transpose, double alpha, double[][] right,
			int rightColumns, double beta, double[][] result) {
		if (right == null || result == null || right.length != result.length)
			throw new IllegalArgumentException("prepared dense batch dimensions do not conform");
		for (int i = 0; i < right.length; i++)
			multiply(transpose, alpha, right[i], rightColumns, beta, result[i]);
	}
	@Override void close();
}
