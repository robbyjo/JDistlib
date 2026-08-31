/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.FloatCsrMatrix;

/**
 * Backend-neutral FP32 BLAS, sparse-BLAS, and reusable factorization surface.
 * Dense matrices use row-major storage. Reduction order and final rounding may
 * differ between providers.
 */
public interface SinglePrecisionLinearAlgebraBackend {
	// BLAS level 1
	default void sscal(int count, float alpha, float[] x, int offset, int stride) {
		CpuAdvancedLinearAlgebra.sscal(count, alpha, x, offset, stride);
	}
	default void scopy(int count, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.scopy(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	default void sswap(int count, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.sswap(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	default float sasum(int count, float[] x, int offset, int stride) {
		return CpuAdvancedLinearAlgebra.sasum(count, x, offset, stride);
	}
	/** Returns the zero-based logical index of the first absolute maximum, or -1 when empty. */
	default int isamax(int count, float[] x, int offset, int stride) {
		return CpuAdvancedLinearAlgebra.isamax(count, x, offset, stride);
	}
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

	// Dense BLAS levels 2 and 3
	default void sgemv(MatrixTranspose transpose, int rows, int columns, float alpha,
			float[] matrix, float[] x, float beta, float[] y) {
		CpuSinglePrecisionLinearAlgebra.sgemv(transpose, rows, columns, alpha,
				matrix, x, beta, y);
	}
	/** Region-aware GEMV with a row-major leading dimension and strided vectors. */
	default void sgemv(MatrixTranspose transpose, int rows, int columns, float alpha,
			float[] matrix, int matrixOffset, int matrixLeadingDimension, float[] x,
			int xOffset, int xStride, float beta, float[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.sgemv(transpose, rows, columns, alpha, matrix,
				matrixOffset, matrixLeadingDimension, x, xOffset, xStride, beta,
				y, yOffset, yStride);
	}
	default void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left,
			float[] right, float beta, float[] result) {
		CpuSinglePrecisionLinearAlgebra.sgemm(leftTranspose, rightTranspose, rows,
				columns, shared, alpha, left, right, beta, result);
	}
	/** Region-aware GEMM with row-major offsets and leading dimensions. */
	default void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left,
			int leftOffset, int leftLeadingDimension, float[] right, int rightOffset,
			int rightLeadingDimension, float beta, float[] result, int resultOffset,
			int resultLeadingDimension) {
		CpuAdvancedLinearAlgebra.sgemm(leftTranspose, rightTranspose, rows, columns,
				shared, alpha, left, leftOffset, leftLeadingDimension, right,
				rightOffset, rightLeadingDimension, beta, result, resultOffset,
				resultLeadingDimension);
	}
	default void sgemmBatched(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared,
			float alpha, float[][] left, float[][] right, float beta, float[][] result) {
		if (left == null || right == null || result == null
				|| left.length != right.length || left.length != result.length)
			throw new IllegalArgumentException("FP32 GEMM batch dimensions do not conform");
		for (int i = 0; i < left.length; i++)
			sgemm(leftTranspose, rightTranspose, rows, columns, shared, alpha,
					left[i], right[i], beta, result[i]);
	}
	default void sger(int rows, int columns, float alpha, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride, float[] matrix) {
		CpuAdvancedLinearAlgebra.sger(rows, columns, alpha, x, xOffset, xStride,
				y, yOffset, yStride, matrix);
	}
	default void ssyr(MatrixTriangle triangle, int dimension, float alpha,
			float[] x, int offset, int stride, float[] matrix) {
		CpuAdvancedLinearAlgebra.ssyr(triangle, dimension, alpha, x, offset, stride, matrix);
	}
	default void ssyr2(MatrixTriangle triangle, int dimension, float alpha,
			float[] x, int xOffset, int xStride, float[] y, int yOffset,
			int yStride, float[] matrix) {
		CpuAdvancedLinearAlgebra.ssyr2(triangle, dimension, alpha, x, xOffset,
				xStride, y, yOffset, yStride, matrix);
	}
	default void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, float beta, float[] result) {
		CpuSinglePrecisionLinearAlgebra.ssyrk(transpose, dimension, shared,
				alpha, matrix, beta, result);
	}
	default void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, int matrixOffset, int matrixLeadingDimension,
			float beta, float[] result, int resultOffset, int resultLeadingDimension) {
		CpuAdvancedLinearAlgebra.ssyrk(transpose, dimension, shared, alpha, matrix,
				matrixOffset, matrixLeadingDimension, beta, result, resultOffset,
				resultLeadingDimension);
	}
	default void ssyr2k(MatrixTriangle triangle, MatrixTranspose transpose,
			int dimension, int shared, float alpha, float[] left, float[] right,
			float beta, float[] result) {
		CpuAdvancedLinearAlgebra.ssyr2k(triangle, transpose, dimension, shared,
				alpha, left, right, beta, result);
	}
	default void ssymm(MatrixSide side, MatrixTriangle triangle, int rows,
			int columns, float alpha, float[] symmetric, float[] right,
			float beta, float[] result) {
		CpuAdvancedLinearAlgebra.ssymm(side, triangle, rows, columns, alpha,
				symmetric, right, beta, result);
	}
	default void strsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		CpuSinglePrecisionLinearAlgebra.strsv(triangle, transpose, diagonal,
				dimension, matrix, vector);
	}
	default void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, float[] right) {
		CpuSinglePrecisionLinearAlgebra.strsm(side, triangle, transpose, diagonal,
				rows, columns, alpha, matrix, right);
	}
	/** Region-aware triangular multi-right-side solve. */
	default void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, int matrixOffset, int matrixLeadingDimension,
			float[] right, int rightOffset, int rightLeadingDimension) {
		CpuAdvancedLinearAlgebra.strsm(side, triangle, transpose, diagonal, rows,
				columns, alpha, matrix, matrixOffset, matrixLeadingDimension, right,
				rightOffset, rightLeadingDimension);
	}

	// Sparse BLAS and prepared operands
	default void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x, float beta,
			float[] y) {
		CpuSinglePrecisionLinearAlgebra.scsrmv(alpha, matrix, x, beta, y);
	}
	default void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		CpuSinglePrecisionLinearAlgebra.scsrmm(alpha, matrix, right, rightColumns,
				beta, result);
	}
	default FloatCsrMatrix scsrgemm(FloatCsrMatrix left, FloatCsrMatrix right) {
		return CpuAdvancedLinearAlgebra.scsrgemm(left, right);
	}
	default void scsrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, FloatCsrMatrix matrix, float[] vector) {
		CpuAdvancedLinearAlgebra.scsrsv(triangle, transpose, diagonal, matrix, vector);
	}
	default PreparedFloatCsrMatrix prepareScsr(FloatCsrMatrix matrix) {
		return CpuPreparedSparse.matrix(this, matrix);
	}
	default PreparedFloatDenseMatrix prepareSge(float[] matrix, int rows, int columns) {
		return CpuAdvancedLinearAlgebra.prepareSge(matrix, rows, columns);
	}

	// Sparse and dense factorizations
	default FloatSparseCholeskyFactor scsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return CpuSparseCholesky.factor(matrix, triangle, ordering);
	}
	default FloatSparseCholeskyFactor scsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle) {
		return scsrpotrf(matrix, triangle, SparseOrdering.MINIMUM_DEGREE);
	}
	default PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return CpuSparseCholesky.prepare(matrix, triangle, ordering);
	}
	default PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle) {
		return prepareScsrpotrf(matrix, triangle, SparseOrdering.MINIMUM_DEGREE);
	}
	default FloatCholeskyFactor spotrf(float[] matrix, int dimension) {
		return CpuSinglePrecisionLinearAlgebra.spotrf(matrix, dimension);
	}
	default FloatCholeskyFactor[] spotrfBatched(float[][] matrices, int dimension) {
		if (matrices == null)
			throw new IllegalArgumentException("FP32 Cholesky batch is required");
		FloatCholeskyFactor[] result = new FloatCholeskyFactor[matrices.length];
		for (int i = 0; i < matrices.length; i++)
			result[i] = spotrf(matrices[i], dimension);
		return result;
	}
	default FloatLuFactor sgetrf(float[] matrix, int dimension) {
		return CpuAdvancedLinearAlgebra.sgetrf(matrix, dimension);
	}
	default FloatLuFactor[] sgetrfBatched(float[][] matrices, int dimension) {
		if (matrices == null) throw new IllegalArgumentException("FP32 LU batch is required");
		FloatLuFactor[] result = new FloatLuFactor[matrices.length];
		for (int i = 0; i < matrices.length; i++) result[i] = sgetrf(matrices[i], dimension);
		return result;
	}
	default FloatSymmetricIndefiniteFactor ssytrf(float[] matrix, int dimension) {
		return CpuAdvancedLinearAlgebra.ssytrf(matrix, dimension);
	}
	default FloatPivotedQrFactor sgeqp3(float[] matrix, int rows, int columns) {
		return CpuSinglePrecisionLinearAlgebra.sgeqp3(matrix, rows, columns);
	}
	default FloatSymmetricEigenDecomposition ssyev(float[] matrix, int dimension) {
		return CpuSinglePrecisionLinearAlgebra.ssyev(matrix, dimension);
	}
	default FloatSymmetricEigenDecomposition ssygvd(float[] matrix, float[] metric,
			int dimension) {
		return CpuAdvancedLinearAlgebra.ssygvd(matrix, metric, dimension);
	}
	default FloatSingularValueDecomposition sgesvd(float[] matrix, int rows,
			int columns) {
		return CpuSinglePrecisionLinearAlgebra.sgesvd(matrix, rows, columns);
	}
}
