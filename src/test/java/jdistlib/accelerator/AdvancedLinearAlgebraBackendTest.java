/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

public class AdvancedLinearAlgebraBackendTest {
	private final CpuComputeBackend backend = new CpuComputeBackend();

	@Test public void remainingLevelOneOperationsHonorStridesAndAliasing() {
		double[] values = {1,99,-3,99,2}; backend.dscal(3,2,values,0,2);
		assertArrayEquals(new double[]{2,99,-6,99,4},values,0);
		assertEquals(12,backend.dasum(3,values,0,2),0); assertEquals(1,backend.idamax(3,values,0,2));
		double[] overlap = {1,2,3,4}; backend.dcopy(3,overlap,0,1,overlap,1,1);
		assertArrayEquals(new double[]{1,1,2,3},overlap,0);
		double[] other = {7,8,9}; backend.dswap(2,overlap,0,2,other,1,1);
		assertArrayEquals(new double[]{8,1,9,3},overlap,0);
		assertArrayEquals(new double[]{7,1,2},other,0);
	}

	@Test public void matrixRegionsAvoidSubmatrixCopies() {
		double[] matrix = {99,99,99, 99,1,2, 99,3,4, 99,99,99};
		double[] x = {99,5,99,6}, y = {99,1,99,2};
		backend.dgemv(MatrixTranspose.NONE,2,2,1,matrix,4,3,x,1,2,1,y,1,2);
		assertArrayEquals(new double[]{99,18,99,41},y,0);
		double[] right = {99,99, 7,8, 9,10, 99,99};
		double[] result = {99,99,99, 99,1,1, 99,1,1, 99,99,99};
		backend.dgemm(MatrixTranspose.NONE,MatrixTranspose.NONE,2,2,2,1,
				matrix,4,3,right,2,2,1,result,4,3);
		assertArrayEquals(new double[]{99,99,99,99,26,29,99,58,65,99,99,99},result,0);
		double[] symmetric = new double[12]; backend.dsyrk(MatrixTranspose.NONE,2,2,1,
				matrix,4,3,0,symmetric,4,3);
		assertEquals(5,symmetric[4],0); assertEquals(11,symmetric[5],0);
		assertEquals(11,symmetric[7],0); assertEquals(25,symmetric[8],0);
		double[] lower = {99,99,99, 99,2,0, 99,1,3};
		double[] rhs = {99,99,99, 99,2,4, 99,7,11};
		backend.dtrsm(MatrixSide.LEFT,MatrixTriangle.LOWER,MatrixTranspose.NONE,
				MatrixDiagonal.NON_UNIT,2,2,1,lower,4,3,rhs,4,3);
		assertEquals(1,rhs[4],0); assertEquals(2,rhs[5],0);
		assertEquals(2,rhs[7],0); assertEquals(3,rhs[8],0);
		float[] floatMatrix={99,99,99,99,1,2,99,3,4},floatRight={99,99,5,6,7,8};
		float[] floatResult={99,99,99,99,0,0,99,0,0};
		backend.sgemm(MatrixTranspose.NONE,MatrixTranspose.NONE,2,2,2,1,
				floatMatrix,4,3,floatRight,2,2,0,floatResult,4,3);
		assertEquals(19,floatResult[4],0);assertEquals(22,floatResult[5],0);
		assertEquals(43,floatResult[7],0);assertEquals(50,floatResult[8],0);
	}

	@Test public void rankUpdatesAndSymmetricMultiplyFollowBlasContracts() {
		double[] ger = new double[6]; backend.dger(2,3,2,new double[]{1,2},0,1,
				new double[]{3,4,5},0,1,ger);
		assertArrayEquals(new double[]{6,8,10,12,16,20},ger,0);
		double[] syr = new double[4]; backend.dsyr(MatrixTriangle.LOWER,2,2,
				new double[]{1,3},0,1,syr);
		assertArrayEquals(new double[]{2,6,6,18},syr,0);
		backend.dsyr2(MatrixTriangle.UPPER,2,1,new double[]{1,2},0,1,
				new double[]{3,4},0,1,syr);
		assertArrayEquals(new double[]{8,16,16,34},syr,0);
		double[] product = new double[4]; backend.dsymm(MatrixSide.LEFT,MatrixTriangle.LOWER,
				2,2,1,new double[]{2,0,1,3},new double[]{1,2,4,5},0,product);
		assertArrayEquals(new double[]{6,9,13,17},product,0);
		double[] rank2k = new double[4]; backend.dsyr2k(MatrixTriangle.LOWER,
				MatrixTranspose.NONE,2,2,1,new double[]{1,2,3,4},
				new double[]{5,6,7,8},0,rank2k);
		assertArrayEquals(new double[]{34,62,62,106},rank2k,0);
	}

	@Test public void luSolvesReportsDeterminantAndSupportsBatches() {
		double[] matrix = {0,2,1, 1,-2,-3, 2,3,1};
		LuFactor factor = backend.dgetrf(matrix,3);
		assertArrayEquals(new double[]{1,2,-1},factor.solve(new double[]{3,0,7}),1e-14);
		assertEquals(-1,factor.determinantSign()); assertEquals(Math.log(7),factor.logAbsDeterminant(),1e-14);
		assertEquals(2,backend.dgetrfBatched(new double[][]{{2,0,0,3},{4,1,2,3}},2).length);
		assertEquals(2,backend.dpotrfBatched(new double[][]{{2,0,0,3},{4,1,1,2}},2).length);
	}

	@Test public void symmetricIndefiniteFactorHandlesOneAndTwoByTwoPivots() {
		SymmetricIndefiniteFactor two = backend.dsytrf(new double[]{0,2,2,0},2);
		assertArrayEquals(new double[]{3,-1},two.solve(new double[]{-2,6}),0);
		assertArrayEquals(new int[]{2,0},two.blockSizes()); assertEquals(-1,two.determinantSign());
		assertEquals(Math.log(4),two.logAbsDeterminant(),0);
		double[] matrix = {4,2,0, 2,-1,1, 0,1,3}, expected = {1,-2,3}, right = new double[3];
		for(int row=0;row<3;row++)for(int column=0;column<3;column++)right[row]+=matrix[row*3+column]*expected[column];
		assertArrayEquals(expected,backend.dsytrf(matrix,3).solve(right),2e-14);
	}

	@Test public void symmetricIndefiniteFactorSolvesMixedBlockCongruences() {
		java.util.Random random=new java.util.Random(20260831L);int n=6;
		for(int sample=0;sample<20;sample++){
			double[]lower=new double[n*n],diagonal=new double[n*n];
			for(int i=0;i<n;i++){lower[i*n+i]=1;for(int j=0;j<i;j++)lower[i*n+j]=(random.nextDouble()-.5)*.4;}
			diagonal[1]=diagonal[n]=1.5;diagonal[n+1]=.2;
			for(int i=2;i<n;i++)diagonal[i*n+i]=(i%2==0?1:-1)*(i+1);
			double[]temporary=new double[n*n],matrix=new double[n*n];
			for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)temporary[i*n+j]+=lower[i*n+k]*diagonal[k*n+j];
			for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)matrix[i*n+j]+=temporary[i*n+k]*lower[j*n+k];
			double[]expected=new double[n],right=new double[n];for(int i=0;i<n;i++)expected[i]=random.nextDouble()-.5;
			for(int i=0;i<n;i++)for(int j=0;j<n;j++)right[i]+=matrix[i*n+j]*expected[j];
			assertArrayEquals(expected,backend.dsytrf(matrix,n).solve(right),2e-12);
		}
	}

	@Test public void generalizedEigenvectorsSatisfyBothMatrices() {
		double[] a = {4,0,0,3}, b = {2,0,0,1};
		SymmetricEigenDecomposition factor = backend.dsygvd(a,b,2);
		assertArrayEquals(new double[]{2,3},factor.eigenvalues(),2e-14);
		double[] vectors = factor.eigenvectors();
		for(int column=0;column<2;column++){
			double norm=0;for(int i=0;i<2;i++)for(int j=0;j<2;j++)norm+=vectors[i*2+column]*b[i*2+j]*vectors[j*2+column];
			assertEquals(1,norm,2e-14);
			for(int row=0;row<2;row++){double left=0,right=0;for(int k=0;k<2;k++){left+=a[row*2+k]*vectors[k*2+column];right+=b[row*2+k]*vectors[k*2+column];}
				assertEquals(left,factor.eigenvalues()[column]*right,2e-14);}
		}
	}

	@Test public void sparseProductsAndTriangularSolvesStaySparse() {
		CsrMatrix left = new CsrMatrix(2,3,new double[]{1,2,3},new int[]{1,3,2},new int[]{1,3,4});
		CsrMatrix right = new CsrMatrix(3,2,new double[]{4,5,6},new int[]{2,1,2},new int[]{1,2,3,4});
		CsrMatrix product = backend.dcsrgemm(left,right);
		assertArrayEquals(new double[]{0,16,15,0},product.toDense(),0);
		CsrMatrix lower = new CsrMatrix(3,3,new double[]{2,1,3,4,-2,1},
				new int[]{1,1,2,1,2,3},new int[]{1,2,4,7});
		double[] rhs = {2,7,-1}; backend.dcsrsv(MatrixTriangle.LOWER,MatrixTranspose.NONE,
				MatrixDiagonal.NON_UNIT,lower,rhs); assertArrayEquals(new double[]{1,2,-1},rhs,0);
		double[] transposeRight = {0,8,-1}; backend.dcsrsv(MatrixTriangle.LOWER,
				MatrixTranspose.TRANSPOSE,MatrixDiagonal.NON_UNIT,lower,transposeRight);
		assertArrayEquals(new double[]{1,2,-1},transposeRight,0);
	}

	@Test public void preparedDenseAndBatchedGemmReuseInputs() {
		try(PreparedDenseMatrix matrix=backend.prepareDge(new double[]{1,2,3,4,5,6},2,3)){
			double[] result=new double[4];matrix.multiply(MatrixTranspose.NONE,1,
					new double[]{1,0,0,1,1,1},2,0,result);
			assertArrayEquals(new double[]{4,5,10,11},result,0);
			double[][] rights={{1,0,0,1,1,1},{0,1,1,0,1,1}},results={new double[4],new double[4]};
			matrix.multiplyBatched(MatrixTranspose.NONE,1,rights,2,0,results);
			assertArrayEquals(new double[]{4,5,10,11},results[0],0);
		}
		double[][] result={new double[4],new double[4]};backend.dgemmBatched(MatrixTranspose.NONE,
				MatrixTranspose.NONE,2,2,2,1,new double[][]{{1,0,0,1},{2,0,0,2}},
				new double[][]{{3,4,5,6},{3,4,5,6}},0,result);
		assertArrayEquals(new double[]{3,4,5,6},result[0],0);assertArrayEquals(new double[]{6,8,10,12},result[1],0);
	}

	@Test public void fp32AdvancedSurfaceMatchesFp64Semantics() {
		FloatLuFactor lu=backend.sgetrf(new float[]{0,2,1,-3},2);
		assertArrayEquals(new float[]{2,-1},lu.solve(new float[]{-2,5}),2e-5f);
		FloatSymmetricIndefiniteFactor ldl=backend.ssytrf(new float[]{0,2,2,0},2);
		assertArrayEquals(new float[]{3,-1},ldl.solve(new float[]{-2,6}),2e-5f);
		assertArrayEquals(new float[]{2,3},backend.ssygvd(new float[]{4,0,0,3},
				new float[]{2,0,0,1},2).eigenvalues(),2e-5f);
		FloatCsrMatrix a=new FloatCsrMatrix(1,2,new float[]{2,3},new int[]{1,2},new int[]{1,3});
		FloatCsrMatrix b=new FloatCsrMatrix(2,1,new float[]{4,5},new int[]{1,1},new int[]{1,2,3});
		assertArrayEquals(new float[]{23},backend.scsrgemm(a,b).values(),0);
		float[]x={1,99,-3};backend.sscal(2,2,x,0,2);assertArrayEquals(new float[]{2,99,-6},x,0);
		assertEquals(1,backend.isamax(2,x,0,2));
		try(PreparedFloatDenseMatrix prepared=backend.prepareSge(new float[]{1,2,3,4},2,2)){
			float[]result=new float[2];prepared.multiply(MatrixTranspose.NONE,1,new float[]{1,1},1,0,result);
			assertArrayEquals(new float[]{3,7},result,0);}
		assertTrue(backend.sgetrfBatched(new float[][]{{1,0,0,1}},2).length==1);
	}
}
