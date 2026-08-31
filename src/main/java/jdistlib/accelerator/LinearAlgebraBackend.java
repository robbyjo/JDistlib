/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;

/**
 * Backend-neutral FP64 BLAS, sparse-BLAS, and reusable factorization surface.
 * Dense matrices use contiguous row-major storage. Reduction order and final
 * rounding may differ between providers.
 */
public interface LinearAlgebraBackend {
	/** Performs {@code y := alpha*x + y} on strided vector regions. */
	default void daxpy(int count, double alpha, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		CpuLinearAlgebra.daxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
	}

	/** Returns the dot product of two strided vector regions. */
	default double ddot(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		return CpuLinearAlgebra.ddot(count, x, xOffset, xStride, y, yOffset, yStride);
	}

	/** Returns the stable Euclidean norm of a strided vector region. */
	default double dnrm2(int count, double[] x, int offset, int stride) {
		return CpuLinearAlgebra.dnrm2(count, x, offset, stride);
	}

	/** Performs row-major {@code y := alpha*op(A)*x + beta*y}. */
	default void dgemv(MatrixTranspose transpose, int rows, int columns, double alpha,
			double[] matrix, double[] x, double beta, double[] y) {
		CpuLinearAlgebra.dgemv(transpose, rows, columns, alpha, matrix, x, beta, y);
	}

	/** Performs row-major {@code C := alpha*op(A)*op(B) + beta*C}. */
	default void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left,
			double[] right, double beta, double[] result) {
		CpuLinearAlgebra.dgemm(leftTranspose, rightTranspose, rows, columns, shared,
				alpha, left, right, beta, result);
	}
	/** Performs full row-major {@code C := alpha*op(A)*op(A)' + beta*C}. */
	default void dsyrk(MatrixTranspose transpose,int dimension,int shared,double alpha,
			double[]matrix,double beta,double[]result){
		CpuLinearAlgebra.dsyrk(transpose,dimension,shared,alpha,matrix,beta,result);
	}
	/** Solves a triangular system in place. */
	default void dtrsv(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,
			int dimension,double[]matrix,double[]vector){
		CpuLinearAlgebra.dtrsv(triangle,transpose,diagonal,dimension,matrix,vector);
	}
	/** Solves a triangular matrix system with row-major multiple right sides in place. */
	default void dtrsm(MatrixSide side,MatrixTriangle triangle,MatrixTranspose transpose,
			MatrixDiagonal diagonal,int rows,int columns,double alpha,double[]matrix,double[]right){
		CpuLinearAlgebra.dtrsm(side,triangle,transpose,diagonal,rows,columns,alpha,matrix,right);
	}

	/** Performs {@code y := alpha*A*x + beta*y} for a CSR matrix. */
	default void dcsrmv(double alpha, CsrMatrix matrix, double[] x, double beta,
			double[] y) {
		CpuLinearAlgebra.dcsrmv(alpha, matrix, x, beta, y);
	}

	/** Performs {@code C := alpha*A*B + beta*C}; dense B and C are row-major. */
	default void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		CpuLinearAlgebra.dcsrmm(alpha, matrix, right, rightColumns, beta, result);
	}

	/** Computes a reusable sparse Cholesky factor from one authoritative triangle. */
	default SparseCholeskyFactor dcsrpotrf(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		return CpuSparseCholesky.factor(matrix, triangle, ordering);
	}
	/** Computes a minimum-degree sparse Cholesky factor from one authoritative triangle. */
	default SparseCholeskyFactor dcsrpotrf(CsrMatrix matrix, MatrixTriangle triangle) {
		return dcsrpotrf(matrix, triangle, SparseOrdering.MINIMUM_DEGREE);
	}

	/** Computes a reusable lower Cholesky factor of a row-major SPD matrix. */
	default CholeskyFactor dpotrf(double[] matrix, int dimension) {
		return CpuLinearAlgebra.dpotrf(matrix, dimension);
	}

	/** Computes a reusable column-pivoted Householder QR factorization. */
	default PivotedQrFactor dgeqp3(double[] matrix, int rows, int columns) {
		return CpuLinearAlgebra.dgeqp3(matrix, rows, columns);
	}

	/** Computes all eigenvalues and eigenvectors of a row-major real symmetric matrix. */
	default SymmetricEigenDecomposition dsyev(double[] matrix, int dimension) {
		return CpuLinearAlgebra.dsyev(matrix, dimension);
	}

	/** Computes a thin singular-value decomposition of a row-major matrix. */
	default SingularValueDecomposition dgesvd(double[] matrix, int rows, int columns) {
		return CpuLinearAlgebra.dgesvd(matrix, rows, columns);
	}
}
