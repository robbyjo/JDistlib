/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;

/**
 * Backend-neutral FP64 BLAS, sparse-BLAS, and reusable factorization surface.
 * Dense matrices use row-major storage. Reduction order and final rounding may
 * differ between providers.
 */
public interface LinearAlgebraBackend {
	// BLAS level 1
	default void dscal(int count, double alpha, double[] x, int offset, int stride) {
		CpuAdvancedLinearAlgebra.dscal(count, alpha, x, offset, stride);
	}
	default void dcopy(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.dcopy(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	default void dswap(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.dswap(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	default double dasum(int count, double[] x, int offset, int stride) {
		return CpuAdvancedLinearAlgebra.dasum(count, x, offset, stride);
	}
	/** Returns the zero-based logical index of the first absolute maximum, or -1 when empty. */
	default int idamax(int count, double[] x, int offset, int stride) {
		return CpuAdvancedLinearAlgebra.idamax(count, x, offset, stride);
	}
	default void daxpy(int count, double alpha, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		CpuLinearAlgebra.daxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
	}
	default double ddot(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		return CpuLinearAlgebra.ddot(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	default double dnrm2(int count, double[] x, int offset, int stride) {
		return CpuLinearAlgebra.dnrm2(count, x, offset, stride);
	}

	// Dense BLAS levels 2 and 3
	default void dgemv(MatrixTranspose transpose, int rows, int columns, double alpha,
			double[] matrix, double[] x, double beta, double[] y) {
		CpuLinearAlgebra.dgemv(transpose, rows, columns, alpha, matrix, x, beta, y);
	}
	/** Region-aware GEMV with a row-major leading dimension and strided vectors. */
	default void dgemv(MatrixTranspose transpose, int rows, int columns, double alpha,
			double[] matrix, int matrixOffset, int matrixLeadingDimension, double[] x,
			int xOffset, int xStride, double beta, double[] y, int yOffset, int yStride) {
		CpuAdvancedLinearAlgebra.dgemv(transpose, rows, columns, alpha, matrix,
				matrixOffset, matrixLeadingDimension, x, xOffset, xStride, beta,
				y, yOffset, yStride);
	}
	default void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left,
			double[] right, double beta, double[] result) {
		CpuLinearAlgebra.dgemm(leftTranspose, rightTranspose, rows, columns,
				shared, alpha, left, right, beta, result);
	}
	/** Region-aware GEMM with row-major offsets and leading dimensions. */
	default void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left,
			int leftOffset, int leftLeadingDimension, double[] right, int rightOffset,
			int rightLeadingDimension, double beta, double[] result, int resultOffset,
			int resultLeadingDimension) {
		CpuAdvancedLinearAlgebra.dgemm(leftTranspose, rightTranspose, rows, columns,
				shared, alpha, left, leftOffset, leftLeadingDimension, right,
				rightOffset, rightLeadingDimension, beta, result, resultOffset,
				resultLeadingDimension);
	}
	default void dgemmBatched(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared,
			double alpha, double[][] left, double[][] right, double beta,
			double[][] result) {
		if (left == null || right == null || result == null
				|| left.length != right.length || left.length != result.length)
			throw new IllegalArgumentException("GEMM batch dimensions do not conform");
		for (int i = 0; i < left.length; i++)
			dgemm(leftTranspose, rightTranspose, rows, columns, shared, alpha,
					left[i], right[i], beta, result[i]);
	}
	default void dger(int rows, int columns, double alpha, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride, double[] matrix) {
		CpuAdvancedLinearAlgebra.dger(rows, columns, alpha, x, xOffset, xStride,
				y, yOffset, yStride, matrix);
	}
	default void dsyr(MatrixTriangle triangle, int dimension, double alpha,
			double[] x, int offset, int stride, double[] matrix) {
		CpuAdvancedLinearAlgebra.dsyr(triangle, dimension, alpha, x, offset, stride, matrix);
	}
	default void dsyr2(MatrixTriangle triangle, int dimension, double alpha,
			double[] x, int xOffset, int xStride, double[] y, int yOffset,
			int yStride, double[] matrix) {
		CpuAdvancedLinearAlgebra.dsyr2(triangle, dimension, alpha, x, xOffset,
				xStride, y, yOffset, yStride, matrix);
	}
	default void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, double beta, double[] result) {
		CpuLinearAlgebra.dsyrk(transpose, dimension, shared, alpha, matrix, beta, result);
	}
	default void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, int matrixOffset, int matrixLeadingDimension,
			double beta, double[] result, int resultOffset, int resultLeadingDimension) {
		CpuAdvancedLinearAlgebra.dsyrk(transpose, dimension, shared, alpha, matrix,
				matrixOffset, matrixLeadingDimension, beta, result, resultOffset,
				resultLeadingDimension);
	}
	default void dsyr2k(MatrixTriangle triangle, MatrixTranspose transpose,
			int dimension, int shared, double alpha, double[] left, double[] right,
			double beta, double[] result) {
		CpuAdvancedLinearAlgebra.dsyr2k(triangle, transpose, dimension, shared,
				alpha, left, right, beta, result);
	}
	default void dsymm(MatrixSide side, MatrixTriangle triangle, int rows,
			int columns, double alpha, double[] symmetric, double[] right,
			double beta, double[] result) {
		CpuAdvancedLinearAlgebra.dsymm(side, triangle, rows, columns, alpha,
				symmetric, right, beta, result);
	}
	default void dtrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		CpuLinearAlgebra.dtrsv(triangle, transpose, diagonal, dimension, matrix, vector);
	}
	default void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, double[] right) {
		CpuLinearAlgebra.dtrsm(side, triangle, transpose, diagonal, rows, columns,
				alpha, matrix, right);
	}
	/** Region-aware triangular multi-right-side solve. */
	default void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, int matrixOffset, int matrixLeadingDimension,
			double[] right, int rightOffset, int rightLeadingDimension) {
		CpuAdvancedLinearAlgebra.dtrsm(side, triangle, transpose, diagonal, rows,
				columns, alpha, matrix, matrixOffset, matrixLeadingDimension, right,
				rightOffset, rightLeadingDimension);
	}

	// Sparse BLAS and prepared operands
	default void dcsrmv(double alpha, CsrMatrix matrix, double[] x, double beta,
			double[] y) {
		CpuLinearAlgebra.dcsrmv(alpha, matrix, x, beta, y);
	}
	default void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		CpuLinearAlgebra.dcsrmm(alpha, matrix, right, rightColumns, beta, result);
	}
	default CsrMatrix dcsrgemm(CsrMatrix left, CsrMatrix right) {
		return CpuAdvancedLinearAlgebra.dcsrgemm(left, right);
	}
	default void dcsrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, CsrMatrix matrix, double[] vector) {
		CpuAdvancedLinearAlgebra.dcsrsv(triangle, transpose, diagonal, matrix, vector);
	}
	default PreparedCsrMatrix prepareDcsr(CsrMatrix matrix) {
		return CpuPreparedSparse.matrix(this, matrix);
	}
	default PreparedDenseMatrix prepareDge(double[] matrix, int rows, int columns) {
		return CpuAdvancedLinearAlgebra.prepareDge(matrix, rows, columns);
	}

	// Sparse and dense factorizations
	default SparseCholeskyFactor dcsrpotrf(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		return CpuSparseCholesky.factor(matrix, triangle, ordering);
	}
	default SparseCholeskyFactor dcsrpotrf(CsrMatrix matrix, MatrixTriangle triangle) {
		return dcsrpotrf(matrix, triangle, SparseOrdering.MINIMUM_DEGREE);
	}
	default PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return CpuSparseCholesky.prepare(matrix, triangle, ordering);
	}
	default PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle) {
		return prepareDcsrpotrf(matrix, triangle, SparseOrdering.MINIMUM_DEGREE);
	}
	default CholeskyFactor dpotrf(double[] matrix, int dimension) {
		return CpuLinearAlgebra.dpotrf(matrix, dimension);
	}
	default CholeskyFactor[] dpotrfBatched(double[][] matrices, int dimension) {
		if (matrices == null) throw new IllegalArgumentException("Cholesky batch is required");
		CholeskyFactor[] result = new CholeskyFactor[matrices.length];
		for (int i = 0; i < matrices.length; i++) result[i] = dpotrf(matrices[i], dimension);
		return result;
	}
	default LuFactor dgetrf(double[] matrix, int dimension) {
		return CpuAdvancedLinearAlgebra.dgetrf(matrix, dimension);
	}
	default LuFactor[] dgetrfBatched(double[][] matrices, int dimension) {
		if (matrices == null) throw new IllegalArgumentException("LU batch is required");
		LuFactor[] result = new LuFactor[matrices.length];
		for (int i = 0; i < matrices.length; i++) result[i] = dgetrf(matrices[i], dimension);
		return result;
	}
	default SymmetricIndefiniteFactor dsytrf(double[] matrix, int dimension) {
		return CpuAdvancedLinearAlgebra.dsytrf(matrix, dimension);
	}
	default PivotedQrFactor dgeqp3(double[] matrix, int rows, int columns) {
		return CpuLinearAlgebra.dgeqp3(matrix, rows, columns);
	}
	default SymmetricEigenDecomposition dsyev(double[] matrix, int dimension) {
		return CpuLinearAlgebra.dsyev(matrix, dimension);
	}
	default SymmetricEigenDecomposition dsygvd(double[] matrix, double[] metric,
			int dimension) {
		return CpuAdvancedLinearAlgebra.dsygvd(matrix, metric, dimension);
	}
	default SingularValueDecomposition dgesvd(double[] matrix, int rows, int columns) {
		return CpuLinearAlgebra.dgesvd(matrix, rows, columns);
	}
}
