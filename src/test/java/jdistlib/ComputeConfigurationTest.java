/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.accelerator.Compute;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeCapabilities;
import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.inference.ComputeBackedLogDensity;
import jdistlib.inference.ComputeNuts;
import jdistlib.inference.ChainResult;
import jdistlib.inference.InferenceCliOptions;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.RunManifest;
import jdistlib.inference.SamplingOptions;
import jdistlib.rng.MersenneTwister;

public class ComputeConfigurationTest {
	@Test public void builderAndCliExposeStrictComputePolicies() {
		InferenceCliOptions parsed = InferenceCliOptions.parse(new String[] {
				"input.jdm", "--compute=cuda", "--nuts-offload", "force", "data.json"});
		SamplingOptions options = parsed.applyTo(SamplingOptions.builder()
				.warmupIterations(1).sampleIterations(1)).build();
		assertEquals(Compute.CUDA, options.computeBackend());
		assertEquals(ComputeNuts.FORCE, options.nutsBackend());
		assertArrayEquals(new String[] {"input.jdm", "data.json"}, parsed.remainingArguments());

		InferenceCliOptions alias = InferenceCliOptions.parse(new String[] {"--gpu-nuts"});
		assertEquals(Compute.GPU, alias.computeBackend());
		assertEquals(ComputeNuts.FORCE, alias.nutsBackend());
	}

	@Test public void forcedNutsRejectsCpuTargetRatherThanSilentlyFallingBack() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(1)
				.sampleIterations(1).backend(Compute.GPU).nutsBackend(ComputeNuts.FORCE).build();
		try {
			new NoUTurnSampler().sample(new TaggedTarget(new CpuComputeBackend()),
					new double[] {0.1}, options, new MersenneTwister(1));
			throw new AssertionError("CPU fallback was accepted");
		} catch (IllegalArgumentException expected) {
			assertTrue(expected.getMessage().contains("forced NUTS acceleration"));
		}
	}

	@Test public void manifestRecordsRequestedAndActualComputeProvenance() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(1)
				.sampleIterations(1).backend(Compute.CUDA).nutsBackend(ComputeNuts.FORCE).build();
		TaggedTarget target = new TaggedTarget(new TaggedBackend("cuda"));
		RunManifest manifest = RunManifest.create(new NoUTurnSampler(), target, "model",
				options, 1L, 2L, 3L);
		assertEquals("cuda", manifest.computePolicy());
		assertEquals("cuda", manifest.computeBackend());
		assertEquals("test-device", manifest.computeDevice());
		assertEquals("force", manifest.nutsOffload());
	}

	@Test public void forcedNutsAcceptsMatchingAcceleratorAwareTarget() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(1)
				.sampleIterations(2).maximumTreeDepth(2).backend(Compute.CUDA)
				.nutsBackend(ComputeNuts.FORCE).build();
		ChainResult result = new NoUTurnSampler().sample(
				new TaggedTarget(new TaggedBackend("cuda")), new double[] {0.1},
				options, new MersenneTwister(2));
		assertEquals(ChainResult.Status.SUCCESS, result.status());
		assertEquals(2, result.size());
	}

	private static final class TaggedTarget implements ComputeBackedLogDensity {
		private final ComputeBackend backend;
		TaggedTarget(ComputeBackend backend) { this.backend = backend; }
		@Override public ComputeBackend computeBackend() { return backend; }
		@Override public double logDensity(double[] state) { return -0.5 * state[0] * state[0]; }
		@Override public double logDensityAndGradient(double[] state, double[] gradient) {
			gradient[0] = -state[0]; return logDensity(state);
		}
	}

	private static final class TaggedBackend implements ComputeBackend {
		private final String id; private final CpuComputeBackend delegate = new CpuComputeBackend();
		TaggedBackend(String id) { this.id = id; }
		@Override public String id() { return id; }
		@Override public boolean available() { return true; }
		@Override public ComputeCapabilities capabilities() {
			return new ComputeCapabilities(id, "test-device", true, false, 1024L);
		}
		@Override public double[] unary(UnaryOperation operation, double[] input) { return delegate.unary(operation, input); }
		@Override public double[] axpy(double alpha, double[] x, double[] y) { return delegate.axpy(alpha, x, y); }
		@Override public double dot(double[] x, double[] y) { return delegate.dot(x, y); }
		@Override public double[][] matrixMultiply(double[][] left, double[][] right) { return delegate.matrixMultiply(left, right); }
		@Override public LogisticRegressionBatchResult logisticRegression(double[][] design, double[] outcomes, double[][] states, double priorPrecision) {
			return delegate.logisticRegression(design, outcomes, states, priorPrecision);
		}
		@Override public jdistlib.accelerator.PreparedLogisticRegression prepareLogisticRegression(double[][] design, double[] outcomes) {
			return delegate.prepareLogisticRegression(design, outcomes);
		}
		@Override public void close() {}
	}
}
