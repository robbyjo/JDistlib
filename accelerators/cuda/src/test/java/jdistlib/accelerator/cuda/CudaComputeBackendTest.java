/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.cuda;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Assume;
import org.junit.Test;

import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.FloatSingularValueDecomposition;
import jdistlib.accelerator.FloatSymmetricEigenDecomposition;
import jdistlib.accelerator.Compute;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeBackends;
import jdistlib.accelerator.ComputeSelection;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.inference.AcceleratedLogisticRegression;
import jdistlib.inference.AdaptiveStaticHamiltonianMonteCarlo;
import jdistlib.inference.AdaptiveStaticHmcOptions;
import jdistlib.inference.AdaptiveStaticHmcResult;
import jdistlib.inference.ComputeNuts;
import jdistlib.inference.SamplingOptions;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

public class CudaComputeBackendTest {
	@Test public void cudaServiceCanBeRequiredBySamplingOptions() {
		CudaComputeBackend probe = new CudaComputeBackend(); Assume.assumeTrue(probe.available()); probe.close();
		SamplingOptions options = SamplingOptions.builder().warmupIterations(1).sampleIterations(1)
				.backend(Compute.CUDA).nutsBackend(ComputeNuts.FORCE).build();
		double[][] design = {{1, -1}, {1, 0}, {1, 1}}; double[] outcomes = {0, 0, 1};
		try (AcceleratedLogisticRegression target = AcceleratedLogisticRegression.forNuts(
				options, design, outcomes, 1.0)) {
			assertEquals("cuda", target.computeBackend().id());
		}
		try (ComputeSelection selection = ComputeBackends.select(Compute.CUDA)) {
			assertEquals("cuda", selection.selectedBackend());
		}
	}
	@Test public void cudaMatchesCpuReference() {
		CudaComputeBackend cuda = new CudaComputeBackend();
		Assume.assumeTrue(cuda.available());
		try {
			CpuComputeBackend cpu = new CpuComputeBackend();
			double[] x = {0.2, -0.7, 1.1, 2.0}, y = {1.0, 2.0, -1.0, 0.5};
			assertArrayEquals(cpu.unary(UnaryOperation.LOGISTIC, x),
					cuda.unary(UnaryOperation.LOGISTIC, x), 1e-14);
			assertArrayEquals(cpu.axpy(0.4, x, y), cuda.axpy(0.4, x, y), 1e-14);
			assertEquals(cpu.dot(x, y), cuda.dot(x, y), 1e-13);
			double[][] a = {{1, 2, 3}, {4, 5, 6}}, b = {{2, 1}, {0, 3}, {-1, 2}};
			assertMatrix(cpu.matrixMultiply(a, b), cuda.matrixMultiply(a, b), 1e-13);
			double[][] scoreDesign = {{1, 2, 3}, {4, 5, 6}, {-1, 0.5, 2}};
			double[][] scoreVectors = {{2, -1, 0.25}, {-3, 4, 1}};
			try (PreparedTransposeProduct expectedProduct = cpu.prepareTransposeProduct(scoreDesign);
					PreparedTransposeProduct actualProduct = cuda.prepareTransposeProduct(scoreDesign)) {
				assertMatrix(expectedProduct.multiply(scoreVectors), actualProduct.multiply(scoreVectors), 1e-12);
			}
			double[][] design = {{1, -1}, {0.5, 2}, {-2, 0.25}, {1.5, 0.3}};
			double[] outcomes = {1, 0, 1, 0};
			double[][] states = {{0.2, -0.3}, {1.0, 0.5}, {-0.7, 0.1}};
			LogisticRegressionBatchResult expected = cpu.logisticRegression(design, outcomes, states, 1.0);
			LogisticRegressionBatchResult actual = cuda.logisticRegression(design, outcomes, states, 1.0);
			assertArrayEquals(expected.logDensities(), actual.logDensities(), 1e-12);
			assertMatrix(expected.gradients(), actual.gradients(), 1e-12);
			assertTrue(cuda.capabilities().doublePrecision());
			assertPortableLinearAlgebra(cpu, cuda, 2e-12);
			assertPortableSinglePrecision(cpu, cuda, 3e-5f);
		} finally { cuda.close(); }
	}
	@Test public void cudaLikelihoodFeedsBatchedAdaptiveStaticHmc() {
		CudaComputeBackend cuda = new CudaComputeBackend(); Assume.assumeTrue(cuda.available());
		double[][] design = new double[64][2]; double[] outcomes = new double[64];
		for (int i = 0; i < design.length; i++) { design[i][0] = 1.0; design[i][1] = (i - 31.5) / 16.0; outcomes[i] = i >= 32 ? 1.0 : 0.0; }
		try (AcceleratedLogisticRegression target = new AcceleratedLogisticRegression(cuda, design, outcomes, 0.5)) {
			AdaptiveStaticHmcResult result = AdaptiveStaticHamiltonianMonteCarlo.sample(target,
					new double[][] {{0, -0.2}, {0, 0.2}, {-0.2, 0}, {0.2, 0}},
					AdaptiveStaticHmcOptions.builder().warmupIterations(20).sampleIterations(20).maximumLeapfrogSteps(4).build(), 810L);
			assertEquals(4, result.chains().length); assertEquals(20, result.chains()[0].size());
			for (int chain = 0; chain < result.chains().length; chain++) assertEquals(jdistlib.inference.ChainResult.Status.SUCCESS, result.chains()[chain].status());
		} finally { cuda.close(); }
	}
	@Test public void cudaRunsFp64AndFp32DecompositionsOnDevice() {
		CudaComputeBackend cuda=new CudaComputeBackend();Assume.assumeTrue(cuda.available());try{
			double[] spd={4,2,2,3};assertArrayEquals(new double[]{2,0,1,Math.sqrt(2)},cuda.dpotrf(spd,2).lower(),2e-13);
			assertArrayEquals(new double[]{1,2},cuda.dgeqp3(new double[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new double[]{3,5,7,9}),3e-13);
			double[] symmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(symmetric,reconstruct(cuda.dsyev(symmetric,3)),4e-12);
			double[] tall={1,2,3,4,5,6,-1,2,0,3,1,2},wide={1,2,3,4,-1,0,2,5};
			assertArrayEquals(tall,reconstruct(cuda.dgesvd(tall,4,3)),8e-12);assertArrayEquals(wide,reconstruct(cuda.dgesvd(wide,2,4)),8e-12);
			float[] fspd={4,2,2,3};assertArrayEquals(new float[]{2,0,1,(float)Math.sqrt(2)},cuda.spotrf(fspd,2).lower(),3e-5f);
			assertArrayEquals(new float[]{1,2},cuda.sgeqp3(new float[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new float[]{3,5,7,9}),8e-5f);
			float[] fsymmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(fsymmetric,reconstruct(cuda.ssyev(fsymmetric,3)),2e-4f);
			float[] ftall={1,2,3,4,5,6,-1,2,0,3,1,2},fwide={1,2,3,4,-1,0,2,5};
			assertArrayEquals(ftall,reconstruct(cuda.sgesvd(ftall,4,3)),4e-4f);assertArrayEquals(fwide,reconstruct(cuda.sgesvd(fwide,2,4)),4e-4f);
			assertTrue(cuda.capabilities().nativeFactorizations());
		}finally{cuda.close();}}
	private static double[] reconstruct(SymmetricEigenDecomposition d){int n=d.dimension();double[]s=d.eigenvalues(),v=d.eigenvectors(),r=new double[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static float[] reconstruct(FloatSymmetricEigenDecomposition d){int n=d.dimension();float[]s=d.eigenvalues(),v=d.eigenvectors(),r=new float[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static double[] reconstruct(SingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();double[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new double[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}
	private static float[] reconstruct(FloatSingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();float[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new float[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}
	private static void assertMatrix(double[][] expected, double[][] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], tolerance);
	}
	private static void assertPortableLinearAlgebra(ComputeBackend cpu,
			ComputeBackend actual, double tolerance) {
		double[] x = {99, 1, 99, -2, 99, 3}, expectedY = {5, 99, 6, 99, 7, 99};
		double[] actualY = expectedY.clone(); cpu.daxpy(3, 0.5, x, 1, 2, expectedY, 0, 2);
		actual.daxpy(3, 0.5, x, 1, 2, actualY, 0, 2); assertArrayEquals(expectedY, actualY, tolerance);
		assertEquals(cpu.ddot(3, x, 1, 2, expectedY, 0, 2),
				actual.ddot(3, x, 1, 2, actualY, 0, 2), tolerance);
		assertEquals(cpu.dnrm2(3, x, 1, 2), actual.dnrm2(3, x, 1, 2), tolerance);
		double[] matrix = {1, 2, 3, 4, 5, 6}, vector = {2, -1}, expectedV = {1, 1, 1}, actualV = expectedV.clone();
		cpu.dgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.2, matrix, vector, 0.3, expectedV);
		actual.dgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.2, matrix, vector, 0.3, actualV);
		assertArrayEquals(expectedV, actualV, tolerance);
		double[] right = {2, 0, -1, 1, 3, 2}, expectedM = {1, 1, 1, 1}, actualM = expectedM.clone();
		cpu.dgemm(MatrixTranspose.NONE, MatrixTranspose.TRANSPOSE, 2, 2, 3,
				0.7, matrix, right, -0.2, expectedM);
		actual.dgemm(MatrixTranspose.NONE, MatrixTranspose.TRANSPOSE, 2, 2, 3,
				0.7, matrix, right, -0.2, actualM); assertArrayEquals(expectedM, actualM, tolerance);
		double[] storedTranspose = {1,4,2,5,3,6}, ordinaryRight = {2,1,0,3,-1,2};
		double[] expectedMt = new double[4], actualMt = new double[4];
		cpu.dgemm(MatrixTranspose.TRANSPOSE, MatrixTranspose.NONE, 2, 2, 3,
				1.0, storedTranspose, ordinaryRight, 0.0, expectedMt);
		actual.dgemm(MatrixTranspose.TRANSPOSE, MatrixTranspose.NONE, 2, 2, 3,
				1.0, storedTranspose, ordinaryRight, 0.0, actualMt); assertArrayEquals(expectedMt, actualMt, tolerance);
		CsrMatrix sparse = new CsrMatrix(3, 3, new double[] {2, -1, 3, 4},
				new int[] {1, 3, 2, 3}, new int[] {1, 3, 4, 5});
		double[] expectedS = {1, 1, 1}, actualS = expectedS.clone();
		cpu.dcsrmv(1.3, sparse, new double[] {3, 5, 7}, 0.2, expectedS);
		actual.dcsrmv(1.3, sparse, new double[] {3, 5, 7}, 0.2, actualS);
		assertArrayEquals(expectedS, actualS, tolerance);
		double[] dense = {1, 2, 3, 4, 5, 6}, expectedSm = new double[6], actualSm = new double[6];
		cpu.dcsrmm(0.8, sparse, dense, 2, 0.0, expectedSm);
		actual.dcsrmm(0.8, sparse, dense, 2, 0.0, actualSm); assertArrayEquals(expectedSm, actualSm, tolerance);
		CsrMatrix empty = new CsrMatrix(2, 3, new double[0], new int[0], new int[] {1,1,1});
		double[] emptyResult = {2,3}; actual.dcsrmv(1.0, empty, new double[3], 0.5, emptyResult);
		assertArrayEquals(new double[] {1,1.5}, emptyResult, 0.0);
		assertTrue(actual.capabilities().denseLinearAlgebra());
		assertTrue(actual.capabilities().sparseLinearAlgebra());
	}
	private static void assertPortableSinglePrecision(ComputeBackend cpu,
			ComputeBackend actual, float tolerance) {
		float[] x = {99, 1, 99, -2, 99, 3}, expectedY = {5, 99, 6, 99, 7, 99};
		float[] actualY = expectedY.clone(); cpu.saxpy(3, 0.5f, x, 1, 2, expectedY, 0, 2);
		actual.saxpy(3, 0.5f, x, 1, 2, actualY, 0, 2); assertArrayEquals(expectedY, actualY, tolerance);
		assertEquals(cpu.sdot(3, x, 1, 2, expectedY, 0, 2),
				actual.sdot(3, x, 1, 2, actualY, 0, 2), tolerance);
		assertEquals(cpu.snrm2(3, x, 1, 2), actual.snrm2(3, x, 1, 2), tolerance);
		float[] matrix = {1, 2, 3, 4, 5, 6}, vector = {2, -1};
		float[] expectedV = {1, 1, 1}, actualV = expectedV.clone();
		cpu.sgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.2f, matrix, vector, 0.3f, expectedV);
		actual.sgemv(MatrixTranspose.TRANSPOSE, 2, 3, 1.2f, matrix, vector, 0.3f, actualV);
		assertArrayEquals(expectedV, actualV, tolerance);
		float[] right = {2, 0, -1, 1, 3, 2}, expectedM = {1, 1, 1, 1}, actualM = expectedM.clone();
		cpu.sgemm(MatrixTranspose.NONE, MatrixTranspose.TRANSPOSE, 2, 2, 3,
				0.7f, matrix, right, -0.2f, expectedM);
		actual.sgemm(MatrixTranspose.NONE, MatrixTranspose.TRANSPOSE, 2, 2, 3,
				0.7f, matrix, right, -0.2f, actualM); assertArrayEquals(expectedM, actualM, tolerance);
		FloatCsrMatrix sparse = new FloatCsrMatrix(3, 3, new float[] {2, -1, 3, 4},
				new int[] {1, 3, 2, 3}, new int[] {1, 3, 4, 5});
		float[] expectedS = {1, 1, 1}, actualS = expectedS.clone();
		cpu.scsrmv(1.3f, sparse, new float[] {3, 5, 7}, 0.2f, expectedS);
		actual.scsrmv(1.3f, sparse, new float[] {3, 5, 7}, 0.2f, actualS);
		assertArrayEquals(expectedS, actualS, tolerance);
		float[] dense = {1, 2, 3, 4, 5, 6}, expectedSm = new float[6], actualSm = new float[6];
		cpu.scsrmm(0.8f, sparse, dense, 2, 0.0f, expectedSm);
		actual.scsrmm(0.8f, sparse, dense, 2, 0.0f, actualSm);
		assertArrayEquals(expectedSm, actualSm, tolerance);
	}
}
