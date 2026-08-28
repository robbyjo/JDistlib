/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.opencl;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Assume;
import org.junit.Test;

import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.UnaryOperation;

public class OpenClComputeBackendTest {
	@Test public void matchesCpuReferenceWhenOpenClIsAvailable() {
		OpenClComputeBackend opencl = new OpenClComputeBackend(); Assume.assumeTrue(opencl.available());
		CpuComputeBackend cpu = new CpuComputeBackend();
		try {
			double[] x = {-2.0, -0.25, 0.5, 3.0}, y = {1.0, 2.0, 3.0, 4.0};
			assertArrayEquals(cpu.unary(UnaryOperation.LOGISTIC, x), opencl.unary(UnaryOperation.LOGISTIC, x), 1e-13);
			assertArrayEquals(cpu.axpy(0.3, x, y), opencl.axpy(0.3, x, y), 1e-13);
			assertEquals(cpu.dot(x, y), opencl.dot(x, y), 1e-13);
			double[][] design = {{1, -1}, {1, 0.5}, {1, 2}}, states = {{0.1, -0.4}, {0.5, 0.2}};
			double[] outcomes = {0, 1, 1};
			LogisticRegressionBatchResult expected = cpu.logisticRegression(design, outcomes, states, 0.2);
			LogisticRegressionBatchResult actual = opencl.logisticRegression(design, outcomes, states, 0.2);
			assertArrayEquals(expected.logDensities(), actual.logDensities(), 1e-12);
			for (int i = 0; i < states.length; i++) assertArrayEquals(expected.gradients()[i], actual.gradients()[i], 1e-12);
		} finally { opencl.close(); }
	}
}
