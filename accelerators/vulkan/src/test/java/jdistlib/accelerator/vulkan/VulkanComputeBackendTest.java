/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.vulkan;

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
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

public class VulkanComputeBackendTest {
	@Test public void vulkanServiceCanBeRequiredExplicitly() {
		VulkanComputeBackend probe = new VulkanComputeBackend();
		Assume.assumeTrue(probe.available()); probe.close();
		try (ComputeSelection selection = ComputeBackends.select(Compute.VULKAN)) {
			assertEquals("vulkan", selection.selectedBackend());
			assertTrue(selection.backend().capabilities().doublePrecision());
		}
	}

	@Test public void vulkanMatchesCpuReference() {
		VulkanComputeBackend vulkan = new VulkanComputeBackend();
		Assume.assumeTrue(vulkan.available()); CpuComputeBackend cpu = new CpuComputeBackend();
		try {
			double[] x = {-2.0, -0.25, 0.5, 3.0}, y = {1.0, 2.0, 3.0, 4.0};
			for (UnaryOperation operation : UnaryOperation.values()) {
				double[] input = operation == UnaryOperation.LOG || operation == UnaryOperation.SQRT
						? new double[] {0.2, 0.5, 1.0, 3.0}
						: operation == UnaryOperation.LOG1P ? new double[] {-0.5, 0.0, 0.5, 3.0} : x;
				assertArrayEquals(cpu.unary(operation, input), vulkan.unary(operation, input), 2e-13);
			}
			assertArrayEquals(cpu.axpy(0.3, x, y), vulkan.axpy(0.3, x, y), 1e-13);
			assertEquals(cpu.dot(x, y), vulkan.dot(x, y), 1e-13);
			double[][] a = {{1, 2, 3}, {4, 5, 6}}, b = {{2, 1}, {0, 3}, {-1, 2}};
			assertMatrix(cpu.matrixMultiply(a, b), vulkan.matrixMultiply(a, b), 1e-13);
			double[][] design = {{1, -1}, {0.5, 2}, {-2, 0.25}, {1.5, 0.3}};
			double[] outcomes = {1, 0, 1, 0};
			double[][] states = {{0.2, -0.3}, {1.0, 0.5}, {-0.7, 0.1}};
			LogisticRegressionBatchResult expected = cpu.logisticRegression(design, outcomes, states, 1.0);
			LogisticRegressionBatchResult actual = vulkan.logisticRegression(design, outcomes, states, 1.0);
			assertArrayEquals(expected.logDensities(), actual.logDensities(), 2e-12);
			assertMatrix(expected.gradients(), actual.gradients(), 2e-12);
			assertPortableLinearAlgebra(cpu, vulkan, 3e-12);
			assertPortableSinglePrecision(cpu, vulkan, 3e-5f);
		} finally { vulkan.close(); }
	}

	@Test public void fp64TranscendentalsCoverWideAndCancellationProneInputs() {
		VulkanComputeBackend vulkan = new VulkanComputeBackend();
		Assume.assumeTrue(vulkan.available()); CpuComputeBackend cpu = new CpuComputeBackend();
		try {
			assertRelative(cpu.unary(UnaryOperation.EXP,
					new double[] {-50, -20, -1, 0, 1, 20, 50}), vulkan.unary(UnaryOperation.EXP,
					new double[] {-50, -20, -1, 0, 1, 20, 50}), 3e-14);
			assertRelative(cpu.unary(UnaryOperation.LOG,
					new double[] {1e-300, 1e-12, 0.5, 1, 2, 1e12, 1e300}), vulkan.unary(UnaryOperation.LOG,
					new double[] {1e-300, 1e-12, 0.5, 1, 2, 1e12, 1e300}), 3e-14);
			assertRelative(cpu.unary(UnaryOperation.LOG1P,
					new double[] {-0.999999, -1e-12, 1e-12, 0.5, 1e10}), vulkan.unary(UnaryOperation.LOG1P,
					new double[] {-0.999999, -1e-12, 1e-12, 0.5, 1e10}), 3e-14);
			assertArrayEquals(cpu.unary(UnaryOperation.TANH,
					new double[] {-30, -2, -0.1, 0, 0.1, 2, 30}), vulkan.unary(UnaryOperation.TANH,
					new double[] {-30, -2, -0.1, 0, 0.1, 2, 30}), 3e-14);
		} finally { vulkan.close(); }
	}
	@Test public void vulkanRunsFp64AndFp32DecompositionsOnDevice(){VulkanComputeBackend backend=new VulkanComputeBackend();Assume.assumeTrue(backend.available());try{double[]spd={4,2,2,3};assertArrayEquals(new double[]{2,0,1,Math.sqrt(2)},backend.dpotrf(spd,2).lower(),2e-13);assertArrayEquals(new double[]{1,2},backend.dgeqp3(new double[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new double[]{3,5,7,9}),3e-13);double[]symmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(symmetric,reconstruct(backend.dsyev(symmetric,3)),4e-12);double[]tall={1,2,3,4,5,6,-1,2,0,3,1,2},wide={1,2,3,4,-1,0,2,5};assertArrayEquals(tall,reconstruct(backend.dgesvd(tall,4,3)),8e-12);assertArrayEquals(wide,reconstruct(backend.dgesvd(wide,2,4)),8e-12);float[]fspd={4,2,2,3};assertArrayEquals(new float[]{2,0,1,(float)Math.sqrt(2)},backend.spotrf(fspd,2).lower(),3e-5f);assertArrayEquals(new float[]{1,2},backend.sgeqp3(new float[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new float[]{3,5,7,9}),8e-5f);float[]fsymmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(fsymmetric,reconstruct(backend.ssyev(fsymmetric,3)),2e-4f);float[]ftall={1,2,3,4,5,6,-1,2,0,3,1,2},fwide={1,2,3,4,-1,0,2,5};assertArrayEquals(ftall,reconstruct(backend.sgesvd(ftall,4,3)),4e-4f);assertArrayEquals(fwide,reconstruct(backend.sgesvd(fwide,2,4)),4e-4f);assertTrue(backend.capabilities().nativeFactorizations());}finally{backend.close();}}
	private static double[] reconstruct(SymmetricEigenDecomposition d){int n=d.dimension();double[]s=d.eigenvalues(),v=d.eigenvectors(),r=new double[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static float[] reconstruct(FloatSymmetricEigenDecomposition d){int n=d.dimension();float[]s=d.eigenvalues(),v=d.eigenvectors(),r=new float[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static double[] reconstruct(SingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();double[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new double[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}
	private static float[] reconstruct(FloatSingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();float[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new float[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}

	private static void assertMatrix(double[][] expected, double[][] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], tolerance);
	}
	private static void assertRelative(double[] expected, double[] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertEquals(expected[i], actual[i],
				tolerance * Math.max(1.0, Math.abs(expected[i])));
	}
	private static void assertPortableLinearAlgebra(ComputeBackend cpu, ComputeBackend actual, double tolerance) {
		double[] matrix = {1,2,3,4,5,6}, vector = {2,-1}, expected = {1,1,1}, observed = expected.clone();
		cpu.dgemv(MatrixTranspose.TRANSPOSE,2,3,1.2,matrix,vector,0.3,expected);
		actual.dgemv(MatrixTranspose.TRANSPOSE,2,3,1.2,matrix,vector,0.3,observed);
		assertArrayEquals(expected,observed,tolerance);
		double[] right={2,0,-1,1,3,2}, expectedDense={1,1,1,1}, observedDense=expectedDense.clone();
		cpu.dgemm(MatrixTranspose.NONE,MatrixTranspose.TRANSPOSE,2,2,3,0.7,matrix,right,-0.2,expectedDense);
		actual.dgemm(MatrixTranspose.NONE,MatrixTranspose.TRANSPOSE,2,2,3,0.7,matrix,right,-0.2,observedDense);
		assertArrayEquals(expectedDense,observedDense,tolerance);
		double[] storedTranspose={1,4,2,5,3,6},ordinaryRight={2,1,0,3,-1,2},expectedTranspose=new double[4],observedTranspose=new double[4];
		cpu.dgemm(MatrixTranspose.TRANSPOSE,MatrixTranspose.NONE,2,2,3,1,storedTranspose,ordinaryRight,0,expectedTranspose);
		actual.dgemm(MatrixTranspose.TRANSPOSE,MatrixTranspose.NONE,2,2,3,1,storedTranspose,ordinaryRight,0,observedTranspose);
		assertArrayEquals(expectedTranspose,observedTranspose,tolerance);
		CsrMatrix sparse=new CsrMatrix(3,3,new double[]{2,-1,3,4},new int[]{1,3,2,3},new int[]{1,3,4,5});
		double[] expectedSparse={1,1,1},observedSparse=expectedSparse.clone();
		cpu.dcsrmv(1.3,sparse,new double[]{3,5,7},0.2,expectedSparse);
		actual.dcsrmv(1.3,sparse,new double[]{3,5,7},0.2,observedSparse);
		assertArrayEquals(expectedSparse,observedSparse,tolerance);
		double[] dense={1,2,3,4,5,6},expectedProduct=new double[6],observedProduct=new double[6];
		cpu.dcsrmm(0.8,sparse,dense,2,0,expectedProduct);actual.dcsrmm(0.8,sparse,dense,2,0,observedProduct);
		assertArrayEquals(expectedProduct,observedProduct,tolerance);
		CsrMatrix empty=new CsrMatrix(2,3,new double[0],new int[0],new int[]{1,1,1});double[] emptyResult={2,3};
		actual.dcsrmv(1,empty,new double[3],0.5,emptyResult);assertArrayEquals(new double[]{1,1.5},emptyResult,0);
		assertTrue(actual.capabilities().denseLinearAlgebra()); assertTrue(actual.capabilities().sparseLinearAlgebra());
	}
	private static void assertPortableSinglePrecision(ComputeBackend cpu, ComputeBackend actual, float tolerance) {
		float[] x={99,1,99,-2,99,3},expectedY={5,99,6,99,7,99},observedY=expectedY.clone();
		cpu.saxpy(3,0.5f,x,1,2,expectedY,0,2);actual.saxpy(3,0.5f,x,1,2,observedY,0,2);
		assertArrayEquals(expectedY,observedY,tolerance);
		assertEquals(cpu.sdot(3,x,1,2,expectedY,0,2),actual.sdot(3,x,1,2,observedY,0,2),tolerance);
		assertEquals(cpu.snrm2(3,x,1,2),actual.snrm2(3,x,1,2),tolerance);
		float[] matrix={1,2,3,4,5,6},vector={2,-1},expected={1,1,1},observed=expected.clone();
		cpu.sgemv(MatrixTranspose.TRANSPOSE,2,3,1.2f,matrix,vector,0.3f,expected);
		actual.sgemv(MatrixTranspose.TRANSPOSE,2,3,1.2f,matrix,vector,0.3f,observed);
		assertArrayEquals(expected,observed,tolerance);
		float[] right={2,0,-1,1,3,2},expectedDense={1,1,1,1},observedDense=expectedDense.clone();
		cpu.sgemm(MatrixTranspose.NONE,MatrixTranspose.TRANSPOSE,2,2,3,0.7f,matrix,right,-0.2f,expectedDense);
		actual.sgemm(MatrixTranspose.NONE,MatrixTranspose.TRANSPOSE,2,2,3,0.7f,matrix,right,-0.2f,observedDense);
		assertArrayEquals(expectedDense,observedDense,tolerance);
		FloatCsrMatrix sparse=new FloatCsrMatrix(3,3,new float[]{2,-1,3,4},new int[]{1,3,2,3},new int[]{1,3,4,5});
		float[] expectedSparse={1,1,1},observedSparse=expectedSparse.clone();
		cpu.scsrmv(1.3f,sparse,new float[]{3,5,7},0.2f,expectedSparse);
		actual.scsrmv(1.3f,sparse,new float[]{3,5,7},0.2f,observedSparse);
		assertArrayEquals(expectedSparse,observedSparse,tolerance);
		float[] dense={1,2,3,4,5,6},expectedProduct=new float[6],observedProduct=new float[6];
		cpu.scsrmm(0.8f,sparse,dense,2,0,expectedProduct);actual.scsrmm(0.8f,sparse,dense,2,0,observedProduct);
		assertArrayEquals(expectedProduct,observedProduct,tolerance);
	}
}
