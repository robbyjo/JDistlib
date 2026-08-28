/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.inference.GaussianSparseCoefficientProposal;
import jdistlib.inference.PortableSparseSubsetCheckpoint;
import jdistlib.inference.ResidualInformedSparseCandidateProposal;
import jdistlib.inference.SparseSubsetCheckpointIO;
import jdistlib.inference.SparseSubsetResult;
import jdistlib.inference.SparseSubsetRjSampler;
import jdistlib.inference.SparseSubsetSamplingOptions;
import jdistlib.inference.SparseSubsetState;
import jdistlib.inference.SparseSubsetSummary;
import jdistlib.inference.SparseSubsetTarget;
import jdistlib.inference.UniformSparseCandidateProposal;
import jdistlib.rng.MersenneTwister;

public class SparseReversibleJumpTest {
	@Test public void sparseStateSupportsTranscriptomeScaleCandidateIndices() {
		String[] candidates = names(17000); SparseSubsetTarget target = new SparseSubsetTarget(new String[] {"alpha"}, candidates, 20,
				(common, active, coefficients) -> -0.5 * common[0] * common[0]);
		SparseSubsetState state = target.state(new int[] {7, 16999}, new double[] {0.2}, new double[] {1.0, -1.0});
		assertEquals(2, state.size()); assertEquals(16999, state.activeCandidate(1)); assertEquals("7,16999", state.modelKey());
		assertTrue(state.active(16999)); assertEquals("gene_7+gene_16999", target.modelName(state));
	}

	@Test public void preparedTransposeProductMatchesReferenceBatch() {
		double[][] matrix = {{1, 2, 3}, {4, 5, 6}}, vectors = {{2, -1}, {-3, 4}};
		try (PreparedTransposeProduct prepared = new CpuComputeBackend().prepareTransposeProduct(matrix)) {
			double[][] actual = prepared.multiply(vectors);
			assertArrayEquals(new double[] {-2, -1, 0}, actual[0], 0.0);
			assertArrayEquals(new double[] {13, 14, 15}, actual[1], 0.0);
		}
	}

	@Test public void informedProposalNormalizesOverInactiveCandidates() {
		SparseSubsetTarget target = target(3, 2);
		SparseSubsetState state = target.state(new int[] {1}, new double[] {0.0}, new double[] {0.25});
		try (ResidualInformedSparseCandidateProposal proposal = new ResidualInformedSparseCandidateProposal(
				new CpuComputeBackend().prepareTransposeProduct(new double[][] {{1, 2, 3}, {-1, 0, 2}}),
				ignored -> new double[] {1.0, -1.0}, 0.3, 0.1)) {
			double total = 0.0; for (int candidate = 0; candidate < 3; candidate++)
				total += Math.exp(proposal.logProbability(candidate, state, target));
			assertEquals(1.0, total, 1e-14);
			assertEquals(Double.NEGATIVE_INFINITY, proposal.logProbability(1, state, target), 0.0);
		}
	}

	@Test public void splitCheckpointResumeIsExactlyEquivalent() throws Exception {
		SparseSubsetTarget target = target(8, 3); SparseSubsetResult full = sampler().sample(target, target.state(new double[] {0.0}), options(1000), new MersenneTwister(42));
		SparseSubsetResult first = sampler().sample(target, target.state(new double[] {0.0}), options(500), new MersenneTwister(42));
		Path directory = Files.createTempDirectory("jdistlib-sparse-rj-"); Path checkpointPath = directory.resolve("chain.checkpoint");
		SparseSubsetCheckpointIO.writeAtomic(checkpointPath, first.checkpoint(), "model-v1", "options-v1");
		PortableSparseSubsetCheckpoint restored = SparseSubsetCheckpointIO.read(checkpointPath, "model-v1", "options-v1");
		SparseSubsetResult second = sampler().resume(target, restored.checkpoint(), options(500));
		assertEquals(SparseSubsetResult.Status.SUCCESS, second.status()); assertEquals(full.checkpoint().completedTransitions(), second.checkpoint().completedTransitions());
		assertEquals(full.checkpoint().retainedDraws(), second.checkpoint().retainedDraws()); assertEquals(full.checkpoint().logJoint(), second.checkpoint().logJoint(), 0.0);
		assertEquals(full.checkpoint().state(), second.checkpoint().state()); assertArrayEquals(full.checkpoint().moveWeights(), second.checkpoint().moveWeights(), 0.0);
		assertArrayEquals(full.checkpoint().inclusionCounts(), second.checkpoint().inclusionCounts());
		int offset = first.size(); assertEquals(full.size() - offset, second.size());
		for (int draw = 0; draw < second.size(); draw++) { assertEquals(full.draw(offset + draw), second.draw(draw)); assertEquals(full.logJointAt(offset + draw), second.logJointAt(draw), 0.0); }
		SparseSubsetSummary summary = new SparseSubsetSummary(target, second.checkpoint()); assertEquals(second.checkpoint().retainedDraws(), summary.retainedDraws());
		assertTrue(summary.toJson(0.0).contains("jdistlib.sparse-rjmcmc-summary/1"));
	}

	@Test public void checkpointDuringWarmupPreservesAdaptiveTrajectory() throws Exception {
		SparseSubsetTarget target = target(8, 3);
		SparseSubsetResult full = sampler().sample(target, target.state(new double[] {0.0}), options(1000), new MersenneTwister(144));
		SparseSubsetResult warmupFragment = sampler().sample(target, target.state(new double[] {0.0}), options(100), new MersenneTwister(144));
		SparseSubsetResult resumed = sampler().resume(target, warmupFragment.checkpoint(), options(900));
		assertEquals(full.checkpoint().state(), resumed.checkpoint().state());
		assertEquals(full.checkpoint().logJoint(), resumed.checkpoint().logJoint(), 0.0);
		assertArrayEquals(full.checkpoint().logScales(), resumed.checkpoint().logScales(), 0.0);
		assertArrayEquals(full.checkpoint().scaleUpdates(), resumed.checkpoint().scaleUpdates());
		assertArrayEquals(full.checkpoint().inclusionCounts(), resumed.checkpoint().inclusionCounts());
	}

	@Test public void checksumAndFingerprintRejectUnsafeResume() throws Exception {
		SparseSubsetTarget target = target(4, 2); SparseSubsetResult result = sampler().sample(target, target.state(new double[] {0.0}), options(50), new MersenneTwister(7));
		Path path = Files.createTempFile("jdistlib-sparse-rj-", ".checkpoint"); SparseSubsetCheckpointIO.writeAtomic(path, result.checkpoint(), "model-a", "options-a");
		try { SparseSubsetCheckpointIO.read(path, "model-b", "options-a"); fail("model mismatch accepted"); } catch (IOException expected) { assertTrue(expected.getMessage().contains("model fingerprint")); }
		byte[] bytes = Files.readAllBytes(path); bytes[bytes.length - 1] ^= 1; Files.write(path, bytes);
		try { SparseSubsetCheckpointIO.read(path, null, null); fail("checksum corruption accepted"); } catch (IOException expected) { assertTrue(expected.getMessage().contains("checksum")); }
	}

	@Test public void resumeRejectsChangedWarmupTarget() {
		SparseSubsetTarget target = target(4, 2);
		SparseSubsetResult result = sampler().sample(target, target.state(new double[] {0.0}), options(100), new MersenneTwister(11));
		try { sampler().resume(target, result.checkpoint(), SparseSubsetSamplingOptions.builder()
				.warmupIterations(50).segmentTransitions(10).thinning(2).build()); fail("changed warmup target accepted"); }
		catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("warmup target")); }
	}

	private static SparseSubsetRjSampler sampler() { return new SparseSubsetRjSampler(new UniformSparseCandidateProposal(), new GaussianSparseCoefficientProposal(0.0, 1.0), 0.15, 0.30); }
	private static SparseSubsetSamplingOptions options(int transitions) { return SparseSubsetSamplingOptions.builder().warmupIterations(200).segmentTransitions(transitions).thinning(2).build(); }
	private static SparseSubsetTarget target(int candidates, int maximum) {
		return new SparseSubsetTarget(new String[] {"alpha"}, names(candidates), maximum, (common, active, coefficients) -> {
			double value = -0.5 * common[0] * common[0] - active.length * Math.log(candidates);
			for (int i = 0; i < active.length; i++) { double centered = coefficients[i] - (active[i] == 1 ? 1.0 : 0.0); value -= 0.5 * centered * centered + 0.5 * Math.log(2.0 * Math.PI); }
			return value;
		});
	}
	private static String[] names(int count) { String[] names = new String[count]; for (int i = 0; i < count; i++) names[i] = "gene_" + i; return names; }
}
