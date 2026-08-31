/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

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
	@Test public void symmetricAndTriangularBlasSupportRemlBuildingBlocks(){double[]a={1,2,3,4,5,6},c=new double[4];backend.dsyrk(MatrixTranspose.NONE,2,3,1,a,0,c);assertArrayEquals(new double[]{14,32,32,77},c,1e-15);double[]lower={2,0,1,3},x={2,7};backend.dtrsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,lower,x);assertArrayEquals(new double[]{1,2},x,1e-15);double[]right={2,4,7,11};backend.dtrsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,lower,right);assertArrayEquals(new double[]{1,2,2,3},right,1e-15);double[]rightSide={4,8,7,11};backend.dtrsm(MatrixSide.RIGHT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,lower,rightSide);assertArrayEquals(new double[]{2.0/3,8.0/3,5.0/3,11.0/3},rightSide,1e-15);try(PreparedCholesky factor=backend.prepareDpotrf(new double[]{4,2,2,3},2)){double[]rhs={1,1,2,1};factor.solveInPlace(rhs,2);assertArrayEquals(new double[]{-0.125,0.125,0.75,0.25},rhs,1e-15);assertEquals(Math.log(8),factor.logDeterminant(),1e-15);}}

	@Test public void sparseCholeskyOrdersFactorsAndSolvesWithoutDenseInput() {
		CsrMatrix matrix = new CsrMatrix(5, 5,
				new double[] {4, 1,4, 1,4, 1,4, 1,1,4},
				new int[] {1, 1,2, 2,3, 3,4, 1,4,5},
				new int[] {1,2,4,6,8,11});
		SparseCholeskyFactor factor = backend.dcsrpotrf(matrix, MatrixTriangle.LOWER);
		assertEquals(5, factor.dimension()); assertTrue(factor.nonzeroCount() < 15);
		assertArrayEquals(new double[] {1,2,3,4,5},
				factor.solve(new double[] {11,12,18,24,25}), 2e-14);
		double[] multiple = {11,25, 12,24, 18,18, 24,12, 25,11};
		factor.solveInPlace(multiple, 2);
		assertArrayEquals(new double[] {1,5,2,4,3,3,4,2,5,1}, multiple, 3e-14);
		double[] dense = {4,1,0,0,1, 1,4,1,0,0, 0,1,4,1,0,
				0,0,1,4,1, 1,0,0,1,4};
		assertEquals(backend.dpotrf(dense, 5).logDeterminant(), factor.logDeterminant(), 2e-14);
		assertEquals(ExecutionKind.PORTABLE_FALLBACK, new StubGpuBackend().plan(
				LinearAlgebraOperation.CSR_POTRF, NumericPrecision.FP64, 5, 10).kind());
	}

	@Test public void sparseCholeskyAcceptsUpperTriangleAndNaturalOrdering() {
		CsrMatrix matrix = new CsrMatrix(5, 5,
				new double[] {4,1,1, 4,1, 4,1, 4,1, 4},
				new int[] {1,2,5, 2,3, 3,4, 4,5, 5},
				new int[] {1,4,6,8,10,11});
		SparseCholeskyFactor factor = backend.dcsrpotrf(matrix, MatrixTriangle.UPPER,
				SparseOrdering.NATURAL);
		assertArrayEquals(new int[] {0,1,2,3,4}, factor.permutation());
		assertArrayEquals(new double[] {1,2,3,4,5},
				factor.solve(new double[] {11,12,18,24,25}), 2e-14);
	}

	@Test(expected = IllegalArgumentException.class)
	public void sparseCholeskyRejectsNonPositiveDefiniteInput() {
		backend.dcsrpotrf(new CsrMatrix(1, 1, new double[] {-1},
				new int[] {1}, new int[] {1,2}), MatrixTriangle.LOWER);
	}

	@Test public void sparseCholeskyMatchesDenseReferenceAcrossSparsePatterns() {
		Random random = new Random(421L); int n = 8;
		for (int sample = 0; sample < 8; sample++) {
			double[] matrix = new double[n * n], rowSums = new double[n];
			for (int row = 1; row < n; row++) for (int column = 0; column < row; column++)
				if (random.nextDouble() < 0.3) {
					double value = random.nextDouble() - 0.5;
					matrix[row * n + column] = value; matrix[column * n + row] = value;
					rowSums[row] += Math.abs(value); rowSums[column] += Math.abs(value);
				}
			for (int row = 0; row < n; row++) matrix[row * n + row] = rowSums[row] + 1.0;
			double[] expected = new double[n], right = new double[n];
			for (int row = 0; row < n; row++) expected[row] = random.nextDouble() - 0.5;
			for (int row = 0; row < n; row++) for (int column = 0; column < n; column++)
				right[row] += matrix[row * n + column] * expected[column];
			CsrMatrix sparse = CsrMatrix.fromDense(n, n, matrix, 0.0);
			for (SparseOrdering ordering : SparseOrdering.values()) {
				SparseCholeskyFactor factor = backend.dcsrpotrf(sparse,
						MatrixTriangle.LOWER, ordering);
				assertArrayEquals(expected, factor.solve(right), 2e-14);
				assertEquals(backend.dpotrf(matrix, n).logDeterminant(),
						factor.logDeterminant(), 2e-14);
			}
		}
	}
	@Test public void preparedCsrAndSparseRefactorizationReuseOwnedStructures() {
		CsrMatrix matrix = new CsrMatrix(5, 5,
				new double[] {4, 1,4, 1,4, 1,4, 1,1,4},
				new int[] {1, 1,2, 2,3, 3,4, 1,4,5},
				new int[] {1,2,4,6,8,11});
		try (PreparedCsrMatrix prepared = backend.prepareDcsr(matrix)) {
			double[] product = new double[5];
			prepared.multiply(1.0, new double[] {1,2,3,4,5}, 0.0, product);
			assertArrayEquals(new double[] {4,9,14,19,25}, product, 0.0);
			double[] multiple = new double[10];
			prepared.multiply(1.0, new double[] {1,5,2,4,3,3,4,2,5,1}, 2,
					0.0, multiple);
			assertArrayEquals(new double[] {4,20,9,21,14,16,19,11,25,11}, multiple, 0.0);
		}
		try (PreparedSparseCholesky prepared = backend.prepareDcsrpotrf(matrix,
				MatrixTriangle.LOWER)) {
			assertEquals(10, prepared.structuralNonzeroCount());
			assertArrayEquals(new double[] {1,2,3,4,5},
					prepared.solve(new double[] {11,12,18,24,25}), 2e-14);
			CsrMatrix changed = new CsrMatrix(5, 5,
					new double[] {5, 1,5, 1,5, 1,5, 1,1,5},
					new int[] {1, 1,2, 2,3, 3,4, 1,4,5},
					new int[] {1,2,4,6,8,11});
			prepared.refactor(changed);
			assertArrayEquals(new double[] {1,2,3,4,5},
					prepared.solve(new double[] {12,14,21,28,30}), 2e-14);
			assertEquals(backend.dpotrf(new double[] {5,1,0,0,1, 1,5,1,0,0,
					0,1,5,1,0, 0,0,1,5,1, 1,0,0,1,5}, 5).logDeterminant(),
					prepared.logDeterminant(), 2e-14);
		}
		assertTrue(backend.capabilities().preparedSparseMatrices());
		assertTrue(backend.capabilities().reusableSparseFactorizations());
	}
	private static final class StubGpuBackend implements ComputeBackend {
		@Override public String id() { return "cuda"; }
		@Override public boolean available() { return true; }
		@Override public ComputeCapabilities capabilities() {
			return new ComputeCapabilities("CUDA", "test", true, false, 1L);
		}
		@Override public void close() {}
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
