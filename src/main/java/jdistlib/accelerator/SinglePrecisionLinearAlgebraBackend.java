/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.FloatCsrMatrix;

/** Backend-neutral FP32 BLAS, sparse-BLAS, and factorization surface. */
public interface SinglePrecisionLinearAlgebraBackend {
	default void saxpy(int count, float alpha, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		CpuSinglePrecisionLinearAlgebra.saxpy(count, alpha, x, xOffset, xStride,
				y, yOffset, yStride);
	}
	default float sdot(int count, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		return CpuSinglePrecisionLinearAlgebra.sdot(count, x, xOffset, xStride,
				y, yOffset, yStride);
	}
	default float snrm2(int count, float[] x, int offset, int stride) {
		return CpuSinglePrecisionLinearAlgebra.snrm2(count, x, offset, stride);
	}
	default void sgemv(MatrixTranspose transpose, int rows, int columns, float alpha,
			float[] matrix, float[] x, float beta, float[] y) {
		CpuSinglePrecisionLinearAlgebra.sgemv(transpose, rows, columns, alpha,
				matrix, x, beta, y);
	}
	default void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left,
			float[] right, float beta, float[] result) {
		CpuSinglePrecisionLinearAlgebra.sgemm(leftTranspose, rightTranspose, rows,
				columns, shared, alpha, left, right, beta, result);
	}
	default void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x, float beta,
			float[] y) {
		CpuSinglePrecisionLinearAlgebra.scsrmv(alpha, matrix, x, beta, y);
	}
	default void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		CpuSinglePrecisionLinearAlgebra.scsrmm(alpha, matrix, right, rightColumns,
				beta, result);
	}
	default FloatCholeskyFactor spotrf(float[] matrix, int dimension) {
		return CpuSinglePrecisionLinearAlgebra.spotrf(matrix, dimension);
	}
	default FloatPivotedQrFactor sgeqp3(float[] matrix, int rows, int columns) {
		return CpuSinglePrecisionLinearAlgebra.sgeqp3(matrix, rows, columns);
	}
	default FloatSymmetricEigenDecomposition ssyev(float[] matrix, int dimension) {
		return CpuSinglePrecisionLinearAlgebra.ssyev(matrix, dimension);
	}
	default FloatSingularValueDecomposition sgesvd(float[] matrix, int rows, int columns) {
		return CpuSinglePrecisionLinearAlgebra.sgesvd(matrix, rows, columns);
	}
}
