/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.matrix.CsrMatrix;

public class LinearAlgebraBackendTest {
	private final CpuComputeBackend backend = new CpuComputeBackend();

	@Test public void levelOneSupportsStridedRegionsAndStableNorms() {
		double[] x = {99, 1, 99, -2, 99, 3}, y = {5, 99, 6, 99, 7, 99};
		assertEquals(14.0, backend.ddot(3, x, 1, 2, y, 0, 2), 0.0);
		backend.daxpy(3, 2.0, x, 1, 2, y, 0, 2);
		assertArrayEquals(new double[] {7, 99, 2, 99, 13, 99}, y, 0.0);
		assertEquals(5.0, backend.dnrm2(2, new double[] {3, 99, 4}, 0, 2), 0.0);
		assertEquals(Math.sqrt(2.0) * 1e200,
				backend.dnrm2(2, new double[] {1e200, 1e200}, 0, 1), 2e184);
	}

	@Test public void denseBlasUsesRowMajorTransposeAndAlphaBetaContracts() {
		double[] matrix = {1, 2, 3, 4, 5, 6};
		double[] y = {1, -1};
		backend.dgemv(MatrixTranspose.NONE, 2, 3, 2.0, matrix,
				new double[] {1, 0, -1}, 0.5, y);
		assertArrayEquals(new double[] {-3.5, -4.5}, y, 1e-15);
		double[] transposed = {1, 1, 1};
		backend.dgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.0, matrix,
				new double[] {2, -1}, 1.0, transposed);
		assertArrayEquals(new double[] {-1, 0, 1}, transposed, 1e-15);

		double[] right = {2, 1, 0, 3, -1, 2}, product = {1, 1, 1, 1};
		backend.dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 2, 2, 3,
				1.0, matrix, right, 2.0, product);
		assertArrayEquals(new double[] {1, 15, 4, 33}, product, 1e-15);
		double[] leftStoredTransposed = {1, 4, 2, 5, 3, 6};
		double[] rightStoredTransposed = {2, 0, -1, 1, 3, 2};
		double[] transposeProduct = new double[4];
		backend.dgemm(MatrixTranspose.TRANSPOSE, MatrixTranspose.TRANSPOSE,
				2, 2, 3, 1.0, leftStoredTransposed, rightStoredTransposed,
				0.0, transposeProduct);
		assertArrayEquals(new double[] {-1, 13, 2, 31}, transposeProduct, 1e-15);
	}

	@Test public void csrBlasSupportsVectorAndDenseRightHandSides() {
		CsrMatrix matrix = new CsrMatrix(3, 3, new double[] {2, -1, 3, 4},
				new int[] {1, 3, 2, 3}, new int[] {1, 3, 4, 5});
		double[] y = {1, 1, 1};
		backend.dcsrmv(2.0, matrix, new double[] {3, 5, 7}, -1.0, y);
		assertArrayEquals(new double[] {-3, 29, 55}, y, 1e-15);
		double[] right = {1, 2, 3, 4, 5, 6}, result = {1, 1, 1, 1, 1, 1};
		backend.dcsrmm(1.0, matrix, right, 2, 0.5, result);
		assertArrayEquals(new double[] {-2.5, -1.5, 9.5, 12.5, 20.5, 24.5}, result, 1e-15);
	}

	@Test public void choleskyFactorSolvesAndReportsLogDeterminant() {
		double[] matrix = {4, 2, 2, 3};
		CholeskyFactor factor = backend.dpotrf(matrix, 2);
		assertArrayEquals(new double[] {2, 0, 1, Math.sqrt(2)}, factor.lower(), 1e-15);
		assertArrayEquals(new double[] {0.125, 0.25}, factor.solve(new double[] {1, 1}), 1e-15);
		assertArrayEquals(new double[] {-0.125, 0.125, 0.75, 0.25},
				factor.solve(new double[] {1, 1, 2, 1}, 2), 1e-15);
		assertEquals(Math.log(8.0), factor.logDeterminant(), 1e-15);
		assertTrue(backend.capabilities().nativeFactorizations());
	}

	@Test public void pivotedQrSolvesLeastSquaresAndDetectsRank() {
		double[] design = {1, 1, 10, 1, 2, 20, 1, 3, 30, 1, 4, 40};
		PivotedQrFactor deficient = backend.dgeqp3(design, 4, 3);
		assertEquals(2, deficient.rank());
		double[] fullRank = {1, 1, 1, 2, 1, 3, 1, 4};
		PivotedQrFactor factor = backend.dgeqp3(fullRank, 4, 2);
		assertEquals(2, factor.rank());
		assertArrayEquals(new double[] {1, 2}, factor.solveLeastSquares(
				new double[] {3, 5, 7, 9}), 2e-14);
	}

	@Test public void symmetricEigenDecompositionReconstructsMatrix() {
		double[] matrix = {4, 1, 1, 1, 3, 0, 1, 0, 2};
		SymmetricEigenDecomposition decomposition = backend.dsyev(matrix, 3);
		double[] values = decomposition.eigenvalues();
		assertTrue(values[0] <= values[1] && values[1] <= values[2]);
		assertArrayEquals(matrix, reconstructEigen(decomposition), 2e-13);
	}

	@Test public void thinSvdReconstructsTallWideAndRankDeficientMatrices() {
		double[][] matrices = {
			{1, 2, 3, 4, 5, 6, -1, 2, 0, 3, 1, 2},
			{1, 2, 3, 4, -1, 0, 2, 5},
			{1, 2, 2, 4, 3, 6}
		};
		int[][] shapes = {{4, 3}, {2, 4}, {3, 2}};
		for (int i = 0; i < matrices.length; i++) {
			SingularValueDecomposition decomposition = backend.dgesvd(
					matrices[i], shapes[i][0], shapes[i][1]);
			assertArrayEquals(matrices[i], reconstructSvd(decomposition), 5e-13);
		}
		assertEquals(1, backend.dgesvd(matrices[2], 3, 2).rank());
	}

	private static double[] reconstructEigen(SymmetricEigenDecomposition decomposition) {
		int n = decomposition.dimension(); double[] values = decomposition.eigenvalues();
		double[] vectors = decomposition.eigenvectors(), result = new double[n * n];
		for (int row = 0; row < n; row++) for (int column = 0; column < n; column++)
			for (int k = 0; k < n; k++) result[row * n + column] +=
					vectors[row * n + k] * values[k] * vectors[column * n + k];
		return result;
	}

	private static double[] reconstructSvd(SingularValueDecomposition decomposition) {
		int rows = decomposition.rows(), columns = decomposition.columns(), count = decomposition.components();
		double[] values = decomposition.singularValues(), left = decomposition.leftSingularVectors();
		double[] right = decomposition.rightSingularVectorsTransposed(), result = new double[rows * columns];
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			for (int k = 0; k < count; k++) result[row * columns + column] +=
					left[row * count + k] * values[k] * right[k * columns + column];
		return result;
	}
}
