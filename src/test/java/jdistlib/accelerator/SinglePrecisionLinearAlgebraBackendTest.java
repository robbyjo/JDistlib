/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import jdistlib.matrix.FloatCsrMatrix;

public class SinglePrecisionLinearAlgebraBackendTest {
	private static final float TOLERANCE = 2e-5f;
	private final CpuComputeBackend backend = new CpuComputeBackend();

	@Test public void levelOneUsesFloatStorageStridesAndStableNorms() {
		float[] x = {99, 1, 99, -2, 99, 3}, y = {5, 99, 6, 99, 7, 99};
		assertEquals(14.0f, backend.sdot(3, x, 1, 2, y, 0, 2), 0.0f);
		backend.saxpy(3, 2.0f, x, 1, 2, y, 0, 2);
		assertArrayEquals(new float[] {7, 99, 2, 99, 13, 99}, y, 0.0f);
		assertEquals(5.0f, backend.snrm2(2, new float[] {3, 99, 4}, 0, 2), 0.0f);
		assertEquals((float) Math.sqrt(2.0) * 1e20f,
				backend.snrm2(2, new float[] {1e20f, 1e20f}, 0, 1), 2e13f);
	}

	@Test public void denseSparseAndFactorizationsMirrorFp64Contracts() {
		float[] matrix = {1, 2, 3, 4, 5, 6};
		float[] vectorResult = {1, 1, 1};
		backend.sgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.2f, matrix,
				new float[] {2, -1}, 0.3f, vectorResult);
		assertArrayEquals(new float[] {-2.1f, -0.9f, 0.3f}, vectorResult, TOLERANCE);

		float[] right = {2, 0, -1, 1, 3, 2}, product = {1, 1, 1, 1};
		backend.sgemm(MatrixTranspose.NONE, MatrixTranspose.TRANSPOSE, 2, 2, 3,
				0.7f, matrix, right, -0.2f, product);
		assertArrayEquals(new float[] {-0.9f, 8.9f, 1.2f, 21.5f}, product, TOLERANCE);

		FloatCsrMatrix sparse = new FloatCsrMatrix(3, 3, new float[] {2, -1, 3, 4},
				new int[] {1, 3, 2, 3}, new int[] {1, 3, 4, 5});
		float[] sparseVector = {1, 1, 1};
		backend.scsrmv(2.0f, sparse, new float[] {3, 5, 7}, -1.0f, sparseVector);
		assertArrayEquals(new float[] {-3, 29, 55}, sparseVector, TOLERANCE);
		float[] dense = {1, 2, 3, 4, 5, 6}, sparseProduct = {1, 1, 1, 1, 1, 1};
		backend.scsrmm(1.0f, sparse, dense, 2, 0.5f, sparseProduct);
		assertArrayEquals(new float[] {-2.5f, -1.5f, 9.5f, 12.5f, 20.5f, 24.5f},
				sparseProduct, TOLERANCE);

		FloatCholeskyFactor cholesky = backend.spotrf(new float[] {4, 2, 2, 3}, 2);
		assertArrayEquals(new float[] {0.125f, 0.25f},
				cholesky.solve(new float[] {1, 1}), TOLERANCE);
		assertArrayEquals(new float[] {-0.125f, 0.125f, 0.75f, 0.25f},
				cholesky.solve(new float[] {1, 1, 2, 1}, 2), TOLERANCE);
		assertEquals((float) Math.log(8.0), cholesky.logDeterminant(), TOLERANCE);

		FloatPivotedQrFactor qr = backend.sgeqp3(
				new float[] {1, 1, 1, 2, 1, 3, 1, 4}, 4, 2);
		assertEquals(2, qr.rank());
		assertArrayEquals(new float[] {1, 2},
				qr.solveLeastSquares(new float[] {3, 5, 7, 9}), TOLERANCE);
	}

	@Test public void emptySparseMatrixOnlyAppliesBeta() {
		FloatCsrMatrix empty = new FloatCsrMatrix(2, 3, new float[0],
				new int[0], new int[] {1, 1, 1});
		float[] result = {2, 3}; backend.scsrmv(1.0f, empty, new float[3], 0.5f, result);
		assertArrayEquals(new float[] {1, 1.5f}, result, 0.0f);
	}

	@Test public void fp32EigenAndSvdReconstructTheirInputs() {
		float[] symmetric = {4, 1, 1, 1, 3, 0, 1, 0, 2};
		assertArrayEquals(symmetric, reconstructEigen(backend.ssyev(symmetric, 3)), 8e-5f);
		float[][] matrices = {{1,2,3,4,5,6,-1,2,0,3,1,2}, {1,2,3,4,-1,0,2,5}, {1,2,2,4,3,6}};
		int[][] shapes = {{4,3},{2,4},{3,2}};
		for (int i = 0; i < matrices.length; i++)
			assertArrayEquals(matrices[i], reconstructSvd(backend.sgesvd(
					matrices[i], shapes[i][0], shapes[i][1])), 2e-4f);
		assertEquals(1, backend.sgesvd(matrices[2], 3, 2).rank());
	}

	private static float[] reconstructEigen(FloatSymmetricEigenDecomposition decomposition) {
		int n=decomposition.dimension();float[]values=decomposition.eigenvalues(),vectors=decomposition.eigenvectors(),result=new float[n*n];
		for(int row=0;row<n;row++)for(int column=0;column<n;column++)for(int k=0;k<n;k++)result[row*n+column]+=vectors[row*n+k]*values[k]*vectors[column*n+k];return result;
	}
	private static float[] reconstructSvd(FloatSingularValueDecomposition decomposition) {
		int rows=decomposition.rows(),columns=decomposition.columns(),count=decomposition.components();float[]values=decomposition.singularValues(),left=decomposition.leftSingularVectors(),right=decomposition.rightSingularVectorsTransposed(),result=new float[rows*columns];
		for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)for(int k=0;k<count;k++)result[row*columns+column]+=left[row*count+k]*values[k]*right[k*columns+column];return result;
	}
}
