/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

import jdistlib.inference.AdaptiveStaticHamiltonianMonteCarlo;
import jdistlib.inference.AdaptiveStaticHmcOptions;
import jdistlib.inference.AdaptiveStaticHmcResult;
import jdistlib.inference.AdjustedMclmcTuner;
import jdistlib.inference.AdjustedMclmcTuningOptions;
import jdistlib.inference.AdjustedMclmcTuningResult;
import jdistlib.inference.ChainCheckpoint;
import jdistlib.inference.ChunkedDrawSink;
import jdistlib.inference.ColumnarDraws;
import jdistlib.inference.CheckpointIO;
import jdistlib.inference.DifferentiableLogDensity;
import jdistlib.inference.IterationStats;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.MappedDrawStore;
import jdistlib.inference.ManyShortChains;
import jdistlib.inference.ManyShortChainsResult;
import jdistlib.inference.MonteCarloError;
import jdistlib.inference.ParetoSmoothedImportanceSampling;
import jdistlib.inference.Pathfinder;
import jdistlib.inference.PathfinderFit;
import jdistlib.inference.PathfinderOptions;
import jdistlib.inference.PortableCheckpoint;
import jdistlib.inference.PrecisionContinuation;
import jdistlib.inference.PrecisionContinuationResult;
import jdistlib.inference.PrecisionGoal;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.SamplerCheckpoint;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.RandomWalkMetropolis;
import jdistlib.inference.SuperchainPlan;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

public class InferenceNextGenerationTest {
	private static final DifferentiableLogDensity STANDARD_NORMAL_2 = new DifferentiableLogDensity() {
		@Override public double logDensity(double[] state) { return -0.5 * (state[0] * state[0] + state[1] * state[1]); }
		@Override public double logDensityAndGradient(double[] state, double[] gradient) { gradient[0] = -state[0]; gradient[1] = -state[1]; return logDensity(state); }
	};

	@Test public void multipathPathfinderProducesWeightedFiniteDraws() {
		PathfinderFit fit = Pathfinder.fit(STANDARD_NORMAL_2, new double[][] {{3, -2}, {-3, 2}},
				PathfinderOptions.builder().paths(2).drawsPerPath(100).resampledDraws(200).build(), new MersenneTwister(123));
		assertEquals(200, fit.draws().length); assertTrue(Double.isFinite(fit.paretoK()));
		double mean0 = 0.0, mean1 = 0.0; for (double[] draw : fit.draws()) { mean0 += draw[0]; mean1 += draw[1]; }
		assertEquals(0.0, mean0 / 200.0, 0.3); assertEquals(0.0, mean1 / 200.0, 0.3);
		double sum = 0.0; for (double logWeight : fit.logWeights()) sum += Math.exp(logWeight); assertEquals(1.0, sum, 1e-12);
	}

	@Test public void paretoSmoothingNormalizesAndReportsTailShape() {
		ParetoSmoothedImportanceSampling.Result result = ParetoSmoothedImportanceSampling.smooth(new double[] {-2, -1, -0.5, 0, 0.2, 0.3, 1, 2, 3, 5});
		double sum = 0.0; for (double value : result.logWeights()) sum += Math.exp(value);
		assertEquals(1.0, sum, 1e-12); assertTrue(Double.isFinite(result.paretoK()));
	}

	@Test public void nestedRHatDetectsPersistentSuperchainBias() {
		double[][] stationary = {{-1, 0, 1, 0}, {1, 0, -1, 0}, {-1, 0, 1, 0}, {1, 0, -1, 0}};
		int[] ids = {0, 0, 1, 1}; assertEquals(1.0, McmcDiagnostics.nestedRHat(stationary, ids), 1e-12);
		double[][] biased = {{-3, -2, -1, -2}, {-1, -2, -3, -2}, {1, 2, 3, 2}, {3, 2, 1, 2}};
		assertTrue(McmcDiagnostics.nestedRHat(biased, ids) > 1.2);
	}

	@Test public void manyShortChainsRetainsPlanAndSeedDeterminism() {
		SuperchainPlan plan = new SuperchainPlan(new double[][] {{-1, 0}, {1, 0}}, 2);
		SamplingOptions options = SamplingOptions.builder().warmupIterations(5).sampleIterations(10).stepSize(0.4).build();
		ManyShortChainsResult first = ManyShortChains.run(new RandomWalkMetropolis(), STANDARD_NORMAL_2, plan, options, 77L, 2);
		ManyShortChainsResult second = ManyShortChains.run(new RandomWalkMetropolis(), STANDARD_NORMAL_2, plan, options, 77L, 2);
		assertEquals(4, first.plan().totalChains()); assertArrayEquals(first.chains()[2].sample(5), second.chains()[2].sample(5), 0.0);
		assertTrue(Double.isFinite(first.nestedRankNormalizedRHat(0)));
	}

	@Test public void adaptiveStaticHmcRunsCheesAndSnaperContracts() {
		for (AdaptiveStaticHmcOptions.Criterion criterion : AdaptiveStaticHmcOptions.Criterion.values()) {
			AdaptiveStaticHmcResult result = AdaptiveStaticHamiltonianMonteCarlo.sample(STANDARD_NORMAL_2,
					new double[][] {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}, AdaptiveStaticHmcOptions.builder()
							.criterion(criterion).warmupIterations(80).sampleIterations(80).maximumLeapfrogSteps(8).build(), 991L);
			assertEquals(4, result.chains().length); assertEquals(80, result.chains()[0].size());
			assertTrue(result.leapfrogSteps() >= 1 && result.leapfrogSteps() <= 8); assertTrue(result.stepSize() > 0.0);
		}
	}

	@Test public void adjustedMclmcTuningIsAuditableAndReturnsSamples() {
		AdjustedMclmcTuningResult result = AdjustedMclmcTuner.tuneAndSample(STANDARD_NORMAL_2, new double[] {0.2, -0.1},
				SamplingOptions.builder().warmupIterations(50).sampleIterations(60).stepSize(0.1).build(),
				AdjustedMclmcTuningOptions.builder().pilotWarmup(20).pilotDraws(20).maximumLeapfrogSteps(4).build(), new MersenneTwister(82));
		assertEquals(60, result.chain().size()); assertEquals(result.candidates().length, result.scores().length);
		assertTrue(result.leapfrogSteps() >= 1); assertTrue(result.stepSize() > 0.0);
	}

	@Test public void portableCheckpointValidatesFingerprintsAndRngState() throws Exception {
		MersenneTwister random = new MersenneTwister(44); random.nextDouble();
		SamplerCheckpoint sampler = new SamplerCheckpoint("nuts", 1, 12, 0.2, 0.1,
				new double[][] {{1, 0}, {0, 2}}, new double[] {1, 2, 3, 4, 0.8}, 0, new double[] {0, 0}, new double[][] {{0, 0}, {0, 0}}, 3.0);
		ChainCheckpoint checkpoint = new ChainCheckpoint(new double[] {1, 2}, -3, 12, random, sampler);
		Path path = Files.createTempFile("jdistlib-checkpoint", ".bin");
		try { CheckpointIO.write(path, checkpoint, "model-a", "options-a"); PortableCheckpoint restored = CheckpointIO.read(path, "model-a", "options-a");
			assertArrayEquals(checkpoint.state(), restored.checkpoint().state(), 0.0); assertEquals(checkpoint.random().nextDouble(), restored.checkpoint().random().nextDouble(), 0.0);
			assertEquals("nuts", restored.checkpoint().samplerCheckpoint().sampler());
			try { CheckpointIO.read(path, "wrong-model", "options-a"); throw new AssertionError("fingerprint mismatch accepted"); } catch (java.io.IOException expected) { assertTrue(expected.getMessage().contains("fingerprint")); }
		} finally { Files.deleteIfExists(path); }
	}

	@Test public void compressedDrawSinkRoundTripsSelectedColumns() throws Exception {
		Path path = Files.createTempFile("jdistlib-draws", ".bin"); IterationStats stats = new IterationStats(true, 0.9, 0.1, 1, 0, false, 0, 1);
		try { try (ChunkedDrawSink sink = new ChunkedDrawSink(path, new int[] {2, 0}, 2)) {
			sink.accept(0, new double[] {1, 2, 3}, -1, stats); sink.accept(1, new double[] {4, 5, 6}, -2, stats); sink.accept(2, new double[] {7, 8, 9}, -3, stats); }
			ColumnarDraws draws = ChunkedDrawSink.read(path); assertEquals(3, draws.size()); assertArrayEquals(new double[] {3, 6, 9}, draws.column(0), 0.0);
			assertArrayEquals(new double[] {1, 4, 7}, draws.column(1), 0.0);
		} finally { Files.deleteIfExists(path); }
	}

	@Test public void memoryMappedDrawStoreRoundTripsWithoutHeapRetention() throws Exception {
		Path path = Files.createTempFile("jdistlib-mapped-draws", ".bin"); IterationStats stats = new IterationStats(true, 0.9, 0.1, 1, 0, false, 0, 1);
		try { try (MappedDrawStore store = new MappedDrawStore(path, new int[] {1}, 3)) { store.accept(4, new double[] {2, 3}, -1, stats); store.accept(5, new double[] {4, 5}, -2, stats); }
			ColumnarDraws draws = MappedDrawStore.read(path); assertEquals(2, draws.size()); assertArrayEquals(new double[] {3, 5}, draws.column(0), 0.0);
		} finally { Files.deleteIfExists(path); }
	}

	@Test public void arbitraryFunctionDiagnosticsWorkWithoutMutatingDraws() {
		double[][] samples = {{-2, 1}, {-1, 2}, {0, 3}, {1, 4}, {2, 5}, {1, 6}, {0, 7}, {-1, 8}};
		double ess = MonteCarloError.functionEss(samples, value -> value[0] * value[0]);
		double indicator = MonteCarloError.indicatorEss(samples, value -> value[1] > 4);
		assertTrue(ess >= 1.0 && ess <= samples.length); assertTrue(indicator >= 1.0 && indicator <= samples.length);
		assertFalse(Double.isNaN(MonteCarloError.functionMcse(samples, value -> value[0] + value[1])));
	}

	@Test public void precisionContinuationResumesNutsWithoutRepeatingWarmup() {
		PrecisionContinuationResult result = PrecisionContinuation.run(new NoUTurnSampler(), STANDARD_NORMAL_2,
				new double[] {0.2, -0.1}, SamplingOptions.builder().warmupIterations(30).sampleIterations(20).maximumTreeDepth(5).build(),
				PrecisionGoal.builder(0).minimumDraws(40).maximumChunks(3).absoluteMcse(100.0).relativeMcse(Double.NaN).build(), 20, new MersenneTwister(908));
		assertTrue(result.goalMet()); assertEquals(2, result.chunks()); assertEquals(40, result.chain().size());
	}
}
