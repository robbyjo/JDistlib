/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class AutoComputeBackendTest {
	@Test public void automaticRoutingKeepsSmallVectorsOnCpuAndMovesLargeVectors() {
		CountingBackend cpu = new CountingBackend("cpu");
		CountingBackend gpu = new CountingBackend("cuda");
		AutoComputeBackend automatic = new AutoComputeBackend(cpu, gpu);
		try {
			automatic.unary(UnaryOperation.EXP,
					new double[AutoComputeBackend.VECTOR_THRESHOLD - 1]);
			automatic.unary(UnaryOperation.EXP,
					new double[AutoComputeBackend.VECTOR_THRESHOLD]);
			assertEquals(1, cpu.unaryCalls); assertEquals(1, gpu.unaryCalls);
		} finally { automatic.close(); }
		assertEquals(1, cpu.closeCalls); assertEquals(1, gpu.closeCalls);
	}

	@Test public void automaticRoutingUsesDenseWorkEstimateForPublicGemm() {
		CountingBackend cpu = new CountingBackend("cpu");
		CountingBackend gpu = new CountingBackend("cuda");
		AutoComputeBackend automatic = new AutoComputeBackend(cpu, gpu);
		try {
			automatic.dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 2, 2, 2,
					1.0, new double[4], new double[4], 0.0, new double[4]);
			automatic.dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 100, 100, 100,
					1.0, new double[10000], new double[10000], 0.0, new double[10000]);
			automatic.sgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 2, 2, 2,
					1.0f, new float[4], new float[4], 0.0f, new float[4]);
			automatic.sgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, 100, 100, 100,
					1.0f, new float[10000], new float[10000], 0.0f, new float[10000]);
			assertEquals(1, cpu.gemmCalls); assertEquals(1, gpu.gemmCalls);
			assertEquals(1, cpu.sgemmCalls); assertEquals(1, gpu.sgemmCalls);
		} finally { automatic.close(); }
	}
	@Test public void automaticRoutingUsesCubicWorkEstimateForDecompositions() {
		CountingBackend cpu=new CountingBackend("cpu"),gpu=new CountingBackend("cuda");
		AutoComputeBackend automatic=new AutoComputeBackend(cpu,gpu);try{
			automatic.dpotrf(new double[]{2,0,0,2},2);
			double[] large=new double[10000];for(int i=0;i<100;i++)large[i*100+i]=2;
			automatic.dpotrf(large,100);assertEquals(1,cpu.choleskyCalls);assertEquals(1,gpu.choleskyCalls);
		}finally{automatic.close();}
	}

	private static final class CountingBackend implements ComputeBackend {
		private final String id; private final CpuComputeBackend delegate = new CpuComputeBackend();
		private int unaryCalls, gemmCalls, sgemmCalls, choleskyCalls, closeCalls;
		CountingBackend(String id) { this.id = id; }
		@Override public String id() { return id; }
		@Override public boolean available() { return true; }
		@Override public ComputeCapabilities capabilities() {
			return new ComputeCapabilities(id, id + "-device", true, false, 1L);
		}
		@Override public double[] unary(UnaryOperation operation, double[] input) { unaryCalls++; return delegate.unary(operation, input); }
		@Override public double[] axpy(double alpha, double[] x, double[] y) { return delegate.axpy(alpha, x, y); }
		@Override public double dot(double[] x, double[] y) { return delegate.dot(x, y); }
		@Override public void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
				int rows, int columns, int shared, double alpha, double[] left,
				double[] right, double beta, double[] result) {
			gemmCalls++; delegate.dgemm(leftTranspose, rightTranspose, rows, columns,
					shared, alpha, left, right, beta, result);
		}
		@Override public void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
				int rows, int columns, int shared, float alpha, float[] left,
				float[] right, float beta, float[] result) {
			sgemmCalls++; delegate.sgemm(leftTranspose, rightTranspose, rows, columns,
					shared, alpha, left, right, beta, result);
		}
		@Override public CholeskyFactor dpotrf(double[] matrix,int dimension){choleskyCalls++;return delegate.dpotrf(matrix,dimension);}
		@Override public double[][] matrixMultiply(double[][] left, double[][] right) { return delegate.matrixMultiply(left, right); }
		@Override public LogisticRegressionBatchResult logisticRegression(double[][] design, double[] outcomes, double[][] states, double priorPrecision) {
			return delegate.logisticRegression(design, outcomes, states, priorPrecision);
		}
		@Override public void close() { closeCalls++; }
	}
}
