/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

import jdistlib.inference.CoordinateInsertionTransformation;
import jdistlib.inference.CoordinateSplitTransformation;
import jdistlib.inference.DimensionMatchingValidator;
import jdistlib.inference.FixedDimensionSamplerRjKernel;
import jdistlib.inference.GaussianRjBirthProposal;
import jdistlib.inference.PortableReversibleJumpCheckpoint;
import jdistlib.inference.ReversibleJumpCheckpointIO;
import jdistlib.inference.ReversibleJumpChains;
import jdistlib.inference.ReversibleJumpDiagnosticReport;
import jdistlib.inference.ReversibleJumpDiagnostics;
import jdistlib.inference.ReversibleJumpExport;
import jdistlib.inference.ReversibleJumpModelSpace;
import jdistlib.inference.ReversibleJumpMove;
import jdistlib.inference.ReversibleJumpProposal;
import jdistlib.inference.ReversibleJumpResult;
import jdistlib.inference.ReversibleJumpSampler;
import jdistlib.inference.ReversibleJumpSamplingOptions;
import jdistlib.inference.ReversibleJumpState;
import jdistlib.inference.ReversibleJumpTarget;
import jdistlib.inference.ReversibleJumpWithinModelTransition;
import jdistlib.inference.RandomWalkMetropolis;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.SubsetBirthMove;
import jdistlib.inference.SubsetDeathMove;
import jdistlib.inference.SubsetSelectionTarget;
import jdistlib.inference.SubsetSwapMove;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

public class ReversibleJumpTest {
	private static final double INCLUDED_PROBABILITY = 0.3;
	private static final GaussianRjBirthProposal STANDARD_NORMAL = new GaussianRjBirthProposal(0.0, 1.0);

	@Test public void coordinateInsertionHasValidDimensionMatchInverseAndJacobian() {
		CoordinateInsertionTransformation transform = new CoordinateInsertionTransformation(2L, 3L, 1);
		assertTrue(DimensionMatchingValidator.validate(transform,
				new ReversibleJumpState(2L, 4.0, 8.0), new double[] {6.0}, 0.0));
		assertTrue(DimensionMatchingValidator.validate(new CoordinateSplitTransformation(4L, 5L, 1),
				new ReversibleJumpState(4L, 2.0, 5.0, 8.0), new double[] {1.5}, 1e-14));
	}

	@Test(expected = IllegalArgumentException.class)
	public void dimensionMatchingRejectsNanTolerance() {
		DimensionMatchingValidator.validate(new CoordinateInsertionTransformation(0L, 1L, 0),
				new ReversibleJumpState(0L), new double[] {1.0}, Double.NaN);
	}

	@Test(expected = IllegalArgumentException.class)
	public void samplingScheduleRejectsIterationOverflow() {
		ReversibleJumpSamplingOptions.builder().warmupIterations(1).sampleIterations(Integer.MAX_VALUE)
				.thinning(2).build();
	}

	@Test public void subsetBirthDeathRatioRetainsAllModelDependentConstants() {
		SubsetSelectionTarget target = exactTarget(); ReversibleJumpState empty = target.state(0L, new double[0], new double[0]);
		ReversibleJumpProposal proposal = new SubsetBirthMove(STANDARD_NORMAL).propose(empty, target, new MersenneTwister(44));
		double logRatio = target.logJoint(proposal.proposedState()) - target.logJoint(empty)
				+ proposal.logReverseDensity() - proposal.logForwardDensity() + proposal.logAbsJacobian();
		assertEquals(Math.log(INCLUDED_PROBABILITY / (1.0 - INCLUDED_PROBABILITY)), logRatio, 1e-12);
		ReversibleJumpProposal reverse = new SubsetDeathMove(STANDARD_NORMAL).propose(proposal.proposedState(), target, new MersenneTwister(45));
		double reverseRatio = target.logJoint(reverse.proposedState()) - target.logJoint(proposal.proposedState())
				+ reverse.logReverseDensity() - reverse.logForwardDensity() + reverse.logAbsJacobian();
		assertEquals(-logRatio, reverseRatio, 1e-12);
	}

	@Test public void subsetSwapHasReciprocalPriorMatchedProposalTerms() {
		SubsetSelectionTarget target = new SubsetSelectionTarget(new String[0], new String[] {"a", "b"},
				(common, active, coefficients) -> {
				double result = -Math.log(4.0); for (double coefficient : coefficients)
					result += -0.5 * coefficient * coefficient - 0.5 * Math.log(2.0 * Math.PI); return result;
			});
		ReversibleJumpState current = target.state(1L, new double[0], new double[] {0.4});
		ReversibleJumpProposal proposal = new SubsetSwapMove(STANDARD_NORMAL).propose(current, target, new MersenneTwister(18));
		double logRatio = target.logJoint(proposal.proposedState()) - target.logJoint(current)
				+ proposal.logReverseDensity() - proposal.logForwardDensity() + proposal.logAbsJacobian();
		assertEquals(0.0, logRatio, 1e-12); assertEquals(2L, proposal.proposedState().modelId());
	}

	@Test public void ordinaryFixedDimensionSamplerCanUpdateWithinOneRjModel() {
		SubsetSelectionTarget target = exactTarget(); ReversibleJumpState state = target.state(1L, new double[0], new double[] {0.0});
		FixedDimensionSamplerRjKernel kernel = new FixedDimensionSamplerRjKernel("metropolis", new RandomWalkMetropolis(),
				SamplingOptions.builder().warmupIterations(0).sampleIterations(1).thinning(1).stepSize(0.5).build());
		ReversibleJumpWithinModelTransition transition = kernel.update(state, target.logJoint(state), target, new MersenneTwister(31), false);
		assertEquals(1L, transition.state().modelId()); assertEquals(1, transition.state().dimension());
		assertTrue(Double.isFinite(transition.logJoint()));
	}

	@Test public void samplerRecoversExactPosteriorModelProbabilityAndDiagnostics() {
		SubsetSelectionTarget target = exactTarget(); ReversibleJumpResult[] chains = new ReversibleJumpResult[3];
		for (int chain = 0; chain < chains.length; chain++) chains[chain] = exactSampler().sample(target,
				target.state(chain % 2, new double[0], chain % 2 == 0 ? new double[0] : new double[] {0.0}),
				ReversibleJumpSamplingOptions.builder().warmupIterations(500).sampleIterations(12000)
						.adaptMoveWeights(false).build(), new MersenneTwister(901 + chain));
		ReversibleJumpDiagnosticReport report = ReversibleJumpDiagnostics.analyze(target, chains);
		assertEquals(INCLUDED_PROBABILITY, report.inclusionProbabilities()[0], 0.025);
		assertTrue(report.inclusionEffectiveSampleSizes()[0] > 100.0); assertTrue(report.modelChanges() > 1000L);
		assertTrue(report.roundTrips() > 100); assertTrue(report.toJson().contains("jdistlib.rjmcmc-diagnostics/1"));
		assertTrue(report.toJson().contains("\"transitions\"")); assertTrue(report.toJson().contains("\"parameters\""));
		assertTrue(ReversibleJumpExport.toTidyCsv(chains[0], target).contains("intercept-only"));
	}

	@Test public void checkpointResumeIsExactAndPortable() throws Exception {
		SubsetSelectionTarget target = exactTarget(); ReversibleJumpSampler sampler = exactSampler();
		ReversibleJumpResult initial = sampler.sample(target, target.state(0L, new double[0], new double[0]),
				ReversibleJumpSamplingOptions.builder().warmupIterations(100).sampleIterations(50).build(), new MersenneTwister(77));
		ReversibleJumpSamplingOptions continuation = ReversibleJumpSamplingOptions.builder().warmupIterations(0).sampleIterations(100).build();
		ReversibleJumpResult first = sampler.resume(target, initial.checkpoint(), continuation);
		ReversibleJumpResult second = sampler.resume(target, initial.checkpoint(), continuation);
		for (int draw = 0; draw < first.size(); draw++) {
			assertEquals(first.draw(draw).modelId(), second.draw(draw).modelId());
			assertArrayEquals(first.draw(draw).parameters(), second.draw(draw).parameters(), 0.0);
			assertEquals(first.logJointAt(draw), second.logJointAt(draw), 0.0);
		}
		Path path = Files.createTempFile("jdistlib-rj", ".checkpoint");
		try {
			ReversibleJumpCheckpointIO.write(path, initial.checkpoint(), "model-v1", "options-v1");
			PortableReversibleJumpCheckpoint restored = ReversibleJumpCheckpointIO.read(path, "model-v1", "options-v1");
			assertEquals(initial.checkpoint().state(), restored.checkpoint().state());
			assertEquals(initial.checkpoint().random().nextDouble(), restored.checkpoint().random().nextDouble(), 0.0);
			try { ReversibleJumpCheckpointIO.read(path, "wrong", "options-v1"); throw new AssertionError("fingerprint mismatch accepted"); }
			catch (java.io.IOException expected) { assertTrue(expected.getMessage().contains("fingerprint")); }
		} finally { Files.deleteIfExists(path); }
	}

	@Test public void parallelChainsAreScheduleIndependentAndSeedDeterministic() {
		SubsetSelectionTarget target = exactTarget(); ReversibleJumpState[] initial = {
			target.state(0L, new double[0], new double[0]), target.state(1L, new double[0], new double[] {0.0})};
		ReversibleJumpSamplingOptions options = ReversibleJumpSamplingOptions.builder().warmupIterations(20)
				.sampleIterations(200).adaptMoveWeights(false).build();
		ReversibleJumpResult[] first = ReversibleJumpChains.parallel(() -> exactSampler(), target, initial, options, 991L, 2);
		ReversibleJumpResult[] second = ReversibleJumpChains.parallel(() -> exactSampler(), target, initial, options, 991L, 1);
		for (int chain = 0; chain < first.length; chain++) for (int draw = 0; draw < first[chain].size(); draw++) {
			assertEquals(first[chain].draw(draw).modelId(), second[chain].draw(draw).modelId());
			assertArrayEquals(first[chain].draw(draw).parameters(), second[chain].draw(draw).parameters(), 0.0);
		}
	}

	@Test public void generalAcceptanceEngineUsesNonzeroJacobian() {
		ReversibleJumpTarget target = new ReversibleJumpTarget() {
			@Override public ReversibleJumpModelSpace modelSpace(long model) { return new ReversibleJumpModelSpace(model, "m" + model, "x"); }
			@Override public double logJoint(ReversibleJumpState state) { return state.modelId() == 0L ? 0.0 : -Math.log(2.0); }
		};
		ReversibleJumpMove up = deterministicMove("up", "down", 0L, 1L, Math.log(2.0));
		ReversibleJumpMove down = deterministicMove("down", "up", 1L, 0L, -Math.log(2.0));
		ReversibleJumpResult result = new ReversibleJumpSampler(new ReversibleJumpMove[] {up, down}, new double[] {1.0, 1.0})
				.sample(target, new ReversibleJumpState(0L, 1.0), ReversibleJumpSamplingOptions.builder()
						.warmupIterations(0).sampleIterations(20).adaptMoveWeights(false).build(), new MersenneTwister(9));
		for (int draw = 0; draw < result.size(); draw++) assertEquals((draw + 1) % 2, result.draw(draw).modelId());
		assertEquals(20L, result.moveAccepts(0) + result.moveAccepts(1));
	}

	private static SubsetSelectionTarget exactTarget() {
		return new SubsetSelectionTarget(new String[0], new String[] {"effect"}, (common, active, coefficients) -> {
			if (active.length == 0) return Math.log(1.0 - INCLUDED_PROBABILITY);
			double coefficient = coefficients[0];
			return Math.log(INCLUDED_PROBABILITY) - 0.5 * coefficient * coefficient - 0.5 * Math.log(2.0 * Math.PI);
		});
	}
	private static ReversibleJumpSampler exactSampler() {
		return new ReversibleJumpSampler(new ReversibleJumpMove[] {new SubsetBirthMove(STANDARD_NORMAL), new SubsetDeathMove(STANDARD_NORMAL)},
				new double[] {1.0, 1.0});
	}
	private static ReversibleJumpMove deterministicMove(final String name, final String reverse,
			final long from, final long to, final double logJacobian) {
		return new ReversibleJumpMove() {
			@Override public String name() { return name; }
			@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) { return state.modelId() == from; }
			@Override public ReversibleJumpProposal propose(ReversibleJumpState state, ReversibleJumpTarget target, RandomEngine random) {
				return ReversibleJumpProposal.valid(new ReversibleJumpState(to, state.parameters()), reverse, 0.0, 0.0, logJacobian);
			}
		};
	}
}
