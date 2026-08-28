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

	private static final class CountingBackend implements ComputeBackend {
		private final String id; private final CpuComputeBackend delegate = new CpuComputeBackend();
		private int unaryCalls, closeCalls;
		CountingBackend(String id) { this.id = id; }
		@Override public String id() { return id; }
		@Override public boolean available() { return true; }
		@Override public ComputeCapabilities capabilities() {
			return new ComputeCapabilities(id, id + "-device", true, false, 1L);
		}
		@Override public double[] unary(UnaryOperation operation, double[] input) { unaryCalls++; return delegate.unary(operation, input); }
		@Override public double[] axpy(double alpha, double[] x, double[] y) { return delegate.axpy(alpha, x, y); }
		@Override public double dot(double[] x, double[] y) { return delegate.dot(x, y); }
		@Override public double[][] matrixMultiply(double[][] left, double[][] right) { return delegate.matrixMultiply(left, right); }
		@Override public LogisticRegressionBatchResult logisticRegression(double[][] design, double[] outcomes, double[][] states, double priorPrecision) {
			return delegate.logisticRegression(design, outcomes, states, priorPrecision);
		}
		@Override public void close() { closeCalls++; }
	}
}
