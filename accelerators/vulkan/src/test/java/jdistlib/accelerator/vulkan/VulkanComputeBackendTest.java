/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.vulkan;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Assume;
import org.junit.Test;

import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.Compute;
import jdistlib.accelerator.ComputeBackends;
import jdistlib.accelerator.ComputeSelection;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.UnaryOperation;

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

	private static void assertMatrix(double[][] expected, double[][] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], tolerance);
	}
	private static void assertRelative(double[] expected, double[] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertEquals(expected[i], actual[i],
				tolerance * Math.max(1.0, Math.abs(expected[i])));
	}
}
