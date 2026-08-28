/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.cuda;

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
import jdistlib.inference.AcceleratedLogisticRegression;
import jdistlib.inference.AdaptiveStaticHamiltonianMonteCarlo;
import jdistlib.inference.AdaptiveStaticHmcOptions;
import jdistlib.inference.AdaptiveStaticHmcResult;
import jdistlib.inference.ComputeNuts;
import jdistlib.inference.SamplingOptions;

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
			double[][] design = {{1, -1}, {0.5, 2}, {-2, 0.25}, {1.5, 0.3}};
			double[] outcomes = {1, 0, 1, 0};
			double[][] states = {{0.2, -0.3}, {1.0, 0.5}, {-0.7, 0.1}};
			LogisticRegressionBatchResult expected = cpu.logisticRegression(design, outcomes, states, 1.0);
			LogisticRegressionBatchResult actual = cuda.logisticRegression(design, outcomes, states, 1.0);
			assertArrayEquals(expected.logDensities(), actual.logDensities(), 1e-12);
			assertMatrix(expected.gradients(), actual.gradients(), 1e-12);
			assertTrue(cuda.capabilities().doublePrecision());
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
	private static void assertMatrix(double[][] expected, double[][] actual, double tolerance) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], tolerance);
	}
}
