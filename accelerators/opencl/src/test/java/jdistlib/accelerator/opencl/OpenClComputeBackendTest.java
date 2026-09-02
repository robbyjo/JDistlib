/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.opencl;

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
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.ComputeBackends;
import jdistlib.accelerator.ComputeSelection;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.MatrixDiagonal;
import jdistlib.accelerator.MatrixSide;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

public class OpenClComputeBackendTest {
	@Test public void openClServiceCanBeRequiredExplicitly() {
		OpenClComputeBackend probe = new OpenClComputeBackend(); Assume.assumeTrue(probe.available()); probe.close();
		try (ComputeSelection selection = ComputeBackends.select(Compute.OPENCL)) {
			assertEquals("opencl", selection.selectedBackend());
			assertEquals(ComputeApi.OPENCL, selection.deviceInfo().api());
			assertTrue(!"unknown".equals(selection.deviceInfo().driverVersion()));
		}
	}
	@Test public void matchesCpuReferenceWhenOpenClIsAvailable() {
		OpenClComputeBackend opencl = new OpenClComputeBackend(); Assume.assumeTrue(opencl.available());
		CpuComputeBackend cpu = new CpuComputeBackend();
		try {
			double[] x = {-2.0, -0.25, 0.5, 3.0}, y = {1.0, 2.0, 3.0, 4.0};
			assertArrayEquals(cpu.unary(UnaryOperation.LOGISTIC, x), opencl.unary(UnaryOperation.LOGISTIC, x), 1e-13);
			assertArrayEquals(cpu.axpy(0.3, x, y), opencl.axpy(0.3, x, y), 1e-13);
			assertEquals(cpu.dot(x, y), opencl.dot(x, y), 1e-13);
			double[][] scoreDesign = {{1, 2, 3}, {4, 5, 6}, {-1, 0.5, 2}};
			double[][] scoreVectors = {{2, -1, 0.25}, {-3, 4, 1}};
			try (PreparedTransposeProduct expectedProduct = cpu.prepareTransposeProduct(scoreDesign);
					PreparedTransposeProduct actualProduct = opencl.prepareTransposeProduct(scoreDesign)) {
				for (int i = 0; i < scoreVectors.length; i++) assertArrayEquals(
						expectedProduct.multiply(scoreVectors)[i], actualProduct.multiply(scoreVectors)[i], 1e-12);
			}
			double[][] design = {{1, -1}, {1, 0.5}, {1, 2}}, states = {{0.1, -0.4}, {0.5, 0.2}};
			double[] outcomes = {0, 1, 1};
			LogisticRegressionBatchResult expected = cpu.logisticRegression(design, outcomes, states, 0.2);
			LogisticRegressionBatchResult actual = opencl.logisticRegression(design, outcomes, states, 0.2);
			assertArrayEquals(expected.logDensities(), actual.logDensities(), 1e-12);
			for (int i = 0; i < states.length; i++) assertArrayEquals(expected.gradients()[i], actual.gradients()[i], 1e-12);
			assertPortableLinearAlgebra(cpu, opencl, 2e-12);
			assertPortableSinglePrecision(cpu, opencl, 3e-5f);
		} finally { opencl.close(); }
	}
	@Test public void openClRunsFp64AndFp32DecompositionsOnDevice(){OpenClComputeBackend backend=new OpenClComputeBackend();Assume.assumeTrue(backend.available());try{double[]spd={4,2,2,3};assertArrayEquals(new double[]{2,0,1,Math.sqrt(2)},backend.dpotrf(spd,2).lower(),2e-13);assertArrayEquals(new double[]{1,2},backend.dgeqp3(new double[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new double[]{3,5,7,9}),3e-13);double[]symmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(symmetric,reconstruct(backend.dsyev(symmetric,3)),4e-12);double[]tall={1,2,3,4,5,6,-1,2,0,3,1,2},wide={1,2,3,4,-1,0,2,5};assertArrayEquals(tall,reconstruct(backend.dgesvd(tall,4,3)),8e-12);assertArrayEquals(wide,reconstruct(backend.dgesvd(wide,2,4)),8e-12);float[]fspd={4,2,2,3};assertArrayEquals(new float[]{2,0,1,(float)Math.sqrt(2)},backend.spotrf(fspd,2).lower(),3e-5f);assertArrayEquals(new float[]{1,2},backend.sgeqp3(new float[]{1,1,1,2,1,3,1,4},4,2).solveLeastSquares(new float[]{3,5,7,9}),8e-5f);float[]fsymmetric={4,1,1,1,3,0,1,0,2};assertArrayEquals(fsymmetric,reconstruct(backend.ssyev(fsymmetric,3)),2e-4f);float[]ftall={1,2,3,4,5,6,-1,2,0,3,1,2},fwide={1,2,3,4,-1,0,2,5};assertArrayEquals(ftall,reconstruct(backend.sgesvd(ftall,4,3)),4e-4f);assertArrayEquals(fwide,reconstruct(backend.sgesvd(fwide,2,4)),4e-4f);assertTrue(backend.capabilities().nativeFactorizations());}finally{backend.close();}}
	private static double[] reconstruct(SymmetricEigenDecomposition d){int n=d.dimension();double[]s=d.eigenvalues(),v=d.eigenvectors(),r=new double[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static float[] reconstruct(FloatSymmetricEigenDecomposition d){int n=d.dimension();float[]s=d.eigenvalues(),v=d.eigenvectors(),r=new float[n*n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)r[i*n+j]+=v[i*n+k]*s[k]*v[j*n+k];return r;}
	private static double[] reconstruct(SingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();double[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new double[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}
	private static float[] reconstruct(FloatSingularValueDecomposition d){int m=d.rows(),n=d.columns(),q=d.components();float[]s=d.singularValues(),u=d.leftSingularVectors(),v=d.rightSingularVectorsTransposed(),r=new float[m*n];for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int k=0;k<q;k++)r[i*n+j]+=u[i*q+k]*s[k]*v[k*n+j];return r;}
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
		double[] expectedRank=new double[4],observedRank=new double[4];cpu.dsyrk(MatrixTranspose.NONE,2,3,1,matrix,0,expectedRank);actual.dsyrk(MatrixTranspose.NONE,2,3,1,matrix,0,observedRank);assertArrayEquals(expectedRank,observedRank,tolerance);
		double[] triangular={2,0,1,3},expectedSolve={2,10},observedSolve=expectedSolve.clone();cpu.dtrsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,triangular,expectedSolve);actual.dtrsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,triangular,observedSolve);assertArrayEquals(expectedSolve,observedSolve,tolerance);
		double[] expectedMulti={2,4,10,14},observedMulti=expectedMulti.clone();cpu.dtrsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,expectedMulti);actual.dtrsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,observedMulti);assertArrayEquals(expectedMulti,observedMulti,tolerance);
		double[] expectedRight={2,10,4,14},observedRight=expectedRight.clone();cpu.dtrsm(MatrixSide.RIGHT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,expectedRight);actual.dtrsm(MatrixSide.RIGHT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,observedRight);assertArrayEquals(expectedRight,observedRight,tolerance);
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
		float[] expectedRank=new float[4],observedRank=new float[4];cpu.ssyrk(MatrixTranspose.NONE,2,3,1,matrix,0,expectedRank);actual.ssyrk(MatrixTranspose.NONE,2,3,1,matrix,0,observedRank);assertArrayEquals(expectedRank,observedRank,tolerance);
		float[] triangular={2,0,1,3},expectedSolve={2,10},observedSolve=expectedSolve.clone();cpu.strsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,triangular,expectedSolve);actual.strsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,triangular,observedSolve);assertArrayEquals(expectedSolve,observedSolve,tolerance);
		float[] expectedMulti={2,4,10,14},observedMulti=expectedMulti.clone();cpu.strsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,expectedMulti);actual.strsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,MatrixDiagonal.NON_UNIT,2,2,1,triangular,observedMulti);assertArrayEquals(expectedMulti,observedMulti,tolerance);
		FloatCsrMatrix sparse=new FloatCsrMatrix(3,3,new float[]{2,-1,3,4},new int[]{1,3,2,3},new int[]{1,3,4,5});
		float[] expectedSparse={1,1,1},observedSparse=expectedSparse.clone();
		cpu.scsrmv(1.3f,sparse,new float[]{3,5,7},0.2f,expectedSparse);
		actual.scsrmv(1.3f,sparse,new float[]{3,5,7},0.2f,observedSparse);
		assertArrayEquals(expectedSparse,observedSparse,tolerance);
		float[] dense={1,2,3,4,5,6},expectedProduct=new float[6],observedProduct=new float[6];
		cpu.scsrmm(0.8f,sparse,dense,2,0,expectedProduct);actual.scsrmm(0.8f,sparse,dense,2,0,observedProduct);
		assertArrayEquals(expectedProduct,observedProduct,tolerance);
	}
	@Test public void preparedDenseMatricesRemainOnOpenClDevice(){OpenClComputeBackend backend=new OpenClComputeBackend();Assume.assumeTrue(backend.available());try{try(jdistlib.accelerator.PreparedDenseMatrix prepared=backend.prepareDge(new double[]{1,2,3,4,5,6},2,3)){double[]result=new double[4];prepared.multiply(MatrixTranspose.NONE,1,new double[]{1,0,0,1,1,1},2,0,result);assertArrayEquals(new double[]{4,5,10,11},result,1e-12);}try(jdistlib.accelerator.PreparedFloatDenseMatrix prepared=backend.prepareSge(new float[]{1,2,3,4},2,2)){float[]result=new float[2];prepared.multiply(MatrixTranspose.NONE,1,new float[]{1,1},1,0,result);assertArrayEquals(new float[]{3,7},result,2e-5f);}assertTrue(backend.capabilities().preparedDenseMatrices());}finally{backend.close();}}
	@Test public void preparedSparseFactorsRemainOnOpenClDevice(){OpenClComputeBackend backend=new OpenClComputeBackend();Assume.assumeTrue(backend.available());try{
		CsrMatrix matrix=new CsrMatrix(2,2,new double[]{4,1,3},new int[]{1,1,2},new int[]{1,2,4});
		try(jdistlib.accelerator.PreparedSparseCholesky factor=backend.prepareDcsrpotrf(matrix,MatrixTriangle.LOWER)){assertArrayEquals(new double[]{1,2},factor.solve(new double[]{6,7}),2e-12);assertEquals(Math.log(11),factor.logDeterminant(),2e-12);boolean rejected=false;try{factor.refactor(new CsrMatrix(2,2,new double[]{1,2,1},new int[]{1,1,2},new int[]{1,2,4}));}catch(IllegalArgumentException expected){rejected=true;}assertTrue(rejected);assertArrayEquals(new double[]{1,2},factor.solve(new double[]{6,7}),2e-12);factor.refactor(new CsrMatrix(2,2,new double[]{5,1,4},new int[]{1,1,2},new int[]{1,2,4}));assertArrayEquals(new double[]{1,2},factor.solve(new double[]{7,9}),2e-12);}
		CsrMatrix fill=new CsrMatrix(3,3,new double[]{4,1,3,1,3},new int[]{1,1,2,1,3},new int[]{1,2,4,6});
		try(jdistlib.accelerator.PreparedSparseCholesky factor=backend.prepareDcsrpotrf(fill,MatrixTriangle.LOWER,jdistlib.accelerator.SparseOrdering.NATURAL)){assertArrayEquals(new double[]{1,4,2,5,3,6},factor.solve(new double[]{9,27,7,19,10,22},2),3e-12);assertEquals(6,factor.factorNonzeroCount());}
		FloatCsrMatrix floats=new FloatCsrMatrix(2,2,new float[]{4,1,3},new int[]{1,1,2},new int[]{1,2,4});
		try(jdistlib.accelerator.PreparedFloatSparseCholesky factor=backend.prepareScsrpotrf(floats,MatrixTriangle.LOWER)){assertArrayEquals(new float[]{1,2},factor.solve(new float[]{6,7}),3e-5f);}
		assertTrue(backend.capabilities().nativeSparseFactorizations());
	}finally{backend.close();}}
}
