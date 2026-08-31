/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import jdistlib.accelerator.CholeskyFactor;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.Compute;
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.ComputeBackends;
import jdistlib.accelerator.ComputeSelection;
import jdistlib.accelerator.ExecutionKind;
import jdistlib.accelerator.LinearAlgebraOperation;
import jdistlib.accelerator.MatrixDiagonal;
import jdistlib.accelerator.MatrixSide;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.accelerator.NumericPrecision;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;
import org.junit.Assume;
import org.junit.Test;

public class NativeCpuComputeBackendTest {
	@Test public void oneMklRunsDenseAndFactorizationOperationsWhenInstalled() {
		try (OneMklComputeBackend backend = new OneMklComputeBackend()) {
			Assume.assumeTrue("system oneMKL is optional: " + backend.unavailableCause(),
					backend.available());
			assertEquals(ComputeApi.ONEMKL, backend.deviceInfo().api());
			double[] product = new double[4];
			backend.dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 2, 2, 3, 1.0,
					new double[] {1, 2, 3, 4, 5, 6}, new double[] {1, 2, 3, 4, 5, 6},
					0.0, product);
			assertArrayEquals(new double[] {22, 28, 49, 64}, product, 1e-12);
			double[] symmetric = new double[4];
			backend.dsyrk(MatrixTranspose.NONE, 2, 3, 1.0,
					new double[] {1, 2, 3, 4, 5, 6}, 0.0, symmetric);
			assertArrayEquals(new double[] {14, 32, 32, 77}, symmetric, 1e-12);
			double[] right = {2, 4, 10, 14};
			backend.dtrsm(MatrixSide.LEFT, MatrixTriangle.LOWER, MatrixTranspose.NONE,
					MatrixDiagonal.NON_UNIT, 2, 2, 1.0, new double[] {2, 0, 1, 3}, right);
			assertArrayEquals(new double[] {1, 2, 3, 4}, right, 1e-12);
			double[] rightSide = {2,10,4,14};
			backend.dtrsm(MatrixSide.RIGHT, MatrixTriangle.LOWER, MatrixTranspose.NONE,
					MatrixDiagonal.NON_UNIT, 2, 2, 1.0, new double[] {2,0,1,3}, rightSide);
			assertArrayEquals(new double[] {-2.0 / 3.0, 10.0 / 3.0,
					-1.0 / 3.0, 14.0 / 3.0}, rightSide, 1e-12);
			CholeskyFactor factor = backend.dpotrf(new double[] {4, 2, 2, 3}, 2);
			assertArrayEquals(new double[] {0.125, 0.25}, factor.solve(new double[] {1, 1}), 1e-12);
			assertTrue(backend.capabilities().nativeFactorizations());
			assertArrayEquals(new double[] {1, 2}, backend.dgeqp3(
					new double[] {1,1,1,2,1,3,1,4}, 4, 2)
					.solveLeastSquares(new double[] {3,5,7,9}), 1e-11);
			double[] source = {4,1,1,1,3,0,1,0,2};
			assertArrayEquals(source, reconstruct(backend.dsyev(source, 3)), 1e-11);
			double[] tall = {1,2,3,4,5,6,-1,2,0,3,1,2};
			assertArrayEquals(tall, reconstruct(backend.dgesvd(tall, 4, 3)), 1e-11);
			float[] floatProduct = new float[4];
			backend.sgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 2, 2, 3, 1.0f,
					new float[] {1,2,3,4,5,6}, new float[] {1,2,3,4,5,6}, 0.0f,
					floatProduct);
			assertArrayEquals(new float[] {22,28,49,64}, floatProduct, 1e-5f);
			float[] floatRight = {2,4,10,14};
			backend.strsm(MatrixSide.LEFT, MatrixTriangle.LOWER, MatrixTranspose.NONE,
					MatrixDiagonal.NON_UNIT, 2, 2, 1.0f, new float[] {2,0,1,3}, floatRight);
			assertArrayEquals(new float[] {1,2,3,4}, floatRight, 1e-5f);
			double[] scaled={1,-2,3};backend.dscal(3,2,scaled,0,1);assertArrayEquals(new double[]{2,-4,6},scaled,0);
			assertEquals(12,backend.dasum(3,scaled,0,1),0);assertEquals(2,backend.idamax(3,scaled,0,1));
			assertArrayEquals(new double[]{1,2,-1},backend.dgetrf(new double[]{0,2,1,1,-2,-3,2,3,1},3).solve(new double[]{3,0,7}),1e-11);
			assertArrayEquals(new double[]{2,3},backend.dsygvd(new double[]{4,0,0,3},new double[]{2,0,0,1},2).eigenvalues(),1e-11);
			try(jdistlib.accelerator.PreparedDenseMatrix prepared=backend.prepareDge(new double[]{1,2,3,4},2,2)){double[]preparedResult=new double[2];prepared.multiply(MatrixTranspose.NONE,1,new double[]{1,1},1,0,preparedResult);assertArrayEquals(new double[]{3,7},preparedResult,1e-12);}
			assertTrue(backend.capabilities().preparedDenseMatrices());assertTrue(backend.capabilities().batchedLinearAlgebra());
		}
	}

	@Test public void serviceLoaderOffersExplicitNativeCpuChoiceWhenInstalled() {
		boolean installed;
		try (OneMklComputeBackend backend = new OneMklComputeBackend()) {
			installed = backend.available();
		}
		Assume.assumeTrue(installed);
		try (ComputeSelection selection = ComputeBackends.select(Compute.ONEMKL)) {
			assertEquals("onemkl", selection.selectedBackend());
			assertEquals(ComputeApi.ONEMKL, selection.backend().deviceInfo().api());
		}
	}

	@Test public void oneMklPardisoReusesAnalysisForFp64AndFp32WhenExported() {
		try (OneMklComputeBackend backend = new OneMklComputeBackend()) {
			Assume.assumeTrue("system oneMKL is optional", backend.available());
			Assume.assumeTrue("installed oneMKL does not export PARDISO diagonal access",
					backend.capabilities().nativeSparseFactorizations());
			assertEquals(ExecutionKind.NATIVE_CPU, backend.plan(
					LinearAlgebraOperation.CSR_ANALYZE, NumericPrecision.FP64, 2, 3).kind());
			CsrMatrix matrix = new CsrMatrix(2, 2, new double[] {4,1,3},
					new int[] {1,1,2}, new int[] {1,2,4});
			try (PreparedSparseCholesky factor = backend.prepareDcsrpotrf(matrix,
					MatrixTriangle.LOWER)) {
				assertArrayEquals(new double[] {1,2}, factor.solve(new double[] {6,7}), 1e-12);
				assertArrayEquals(new double[] {1,3,2,4},
						factor.solve(new double[] {6,16,7,15}, 2), 1e-12);
				assertEquals(Math.log(11.0), factor.logDeterminant(), 1e-12);
				assertEquals(0, factor.permutation().length);
				factor.refactor(new CsrMatrix(2, 2, new double[] {5,1,4},
						new int[] {1,1,2}, new int[] {1,2,4}));
				assertArrayEquals(new double[] {1,2}, factor.solve(new double[] {7,9}), 1e-12);
				assertEquals(Math.log(19.0), factor.logDeterminant(), 1e-12);
			}
			FloatCsrMatrix floats = new FloatCsrMatrix(2, 2, new float[] {4,1,3},
					new int[] {1,1,2}, new int[] {1,2,4});
			try (PreparedFloatSparseCholesky factor = backend.prepareScsrpotrf(floats,
					MatrixTriangle.LOWER)) {
				assertArrayEquals(new float[] {1,2}, factor.solve(new float[] {6,7}), 2e-5f);
				assertEquals((float) Math.log(11.0), factor.logDeterminant(), 2e-5f);
			}
		}
	}

	@Test public void absentOpenBlasIsReportedWithoutBreakingCoreDiscovery() {
		try (OpenBlasComputeBackend backend = new OpenBlasComputeBackend()) {
			if (!backend.available()) assertTrue(backend.unavailableCause() != null);
			else assertTrue(!backend.capabilities().nativeSparseFactorizations());
		}
		assertTrue(Arrays.asList(Compute.values()).contains(Compute.OPENBLAS));
	}
	private static double[] reconstruct(SymmetricEigenDecomposition decomposition) {
		int n = decomposition.dimension(); double[] values = decomposition.eigenvalues();
		double[] vectors = decomposition.eigenvectors(), result = new double[n * n];
		for (int row = 0; row < n; row++) for (int column = 0; column < n; column++)
			for (int component = 0; component < n; component++) result[row * n + column]
					+= vectors[row * n + component] * values[component]
					* vectors[column * n + component];
		return result;
	}
	private static double[] reconstruct(SingularValueDecomposition decomposition) {
		int rows = decomposition.rows(), columns = decomposition.columns();
		int count = decomposition.components(); double[] values = decomposition.singularValues();
		double[] left = decomposition.leftSingularVectors();
		double[] right = decomposition.rightSingularVectorsTransposed();
		double[] result = new double[rows * columns];
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			for (int component = 0; component < count; component++) result[row * columns + column]
					+= left[row * count + component] * values[component]
					* right[component * columns + column];
		return result;
	}
}
