/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP32 dense matrix handle; providers may retain storage on device. */
public interface PreparedFloatDenseMatrix extends AutoCloseable {
	int rows();
	int columns();
	void multiply(MatrixTranspose transpose, float alpha, float[] right, int rightColumns,
			float beta, float[] result);
	default void multiplyBatched(MatrixTranspose transpose, float alpha, float[][] right,
			int rightColumns, float beta, float[][] result) {
		if (right == null || result == null || right.length != result.length)
			throw new IllegalArgumentException("prepared FP32 dense batch dimensions do not conform");
		for (int i = 0; i < right.length; i++)
			multiply(transpose, alpha, right[i], rightColumns, beta, result[i]);
	}
	@Override void close();
}
