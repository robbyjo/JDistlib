/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.AdjustedMicrocanonicalLangevin;
import jdistlib.inference.BarkerGradientSampler;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.DifferentiableLogDensity;
import jdistlib.inference.EllipticalSliceSampler;
import jdistlib.inference.EvaluationCounter;
import jdistlib.inference.Fit;
import jdistlib.inference.GaussianReference;
import jdistlib.inference.LbfgsOptimizer;
import jdistlib.inference.MetropolisAdjustedLangevin;
import jdistlib.inference.MetricConfiguration;
import jdistlib.inference.MonteCarloError;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.OptimizationResult;
import jdistlib.inference.ParallelTempering;
import jdistlib.inference.ParallelTemperingResult;
import jdistlib.inference.PathfinderInitializer;
import jdistlib.inference.PathfinderResult;
import jdistlib.inference.Inference;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.TemperedLogDensity;
import jdistlib.inference.WarmupSchedule;
import jdistlib.inference.WarmupTrace;
import jdistlib.rng.MersenneTwister;

public class InferenceModernizationTest {
	private static final DifferentiableLogDensity STANDARD_NORMAL =
			new DifferentiableLogDensity() {
		@Override public double logDensityAndGradient(double[] state, double[] gradient) {
			double value = 0.0;
			for (int i = 0; i < state.length; i++) {
				value -= 0.5 * state[i] * state[i]; gradient[i] = -state[i];
			}
			return value;
		}
	};

	@Test public void stanWarmupHasFastSlowFastPhases() {
		WarmupSchedule.Resolved schedule = WarmupSchedule.stanDefault().resolve(1000);
		assertEquals(WarmupSchedule.Phase.INITIAL_FAST, schedule.phase(0));
		assertEquals(WarmupSchedule.Phase.SLOW, schedule.phase(75));
		assertEquals(WarmupSchedule.Phase.FINAL_FAST, schedule.phase(950));
		assertTrue(schedule.slowWindowEnds().length >= 3);
	}

	@Test public void nutsCheckpointResumesBitForBit() {
		SamplingOptions forty = SamplingOptions.builder().warmupIterations(30)
				.sampleIterations(40).maximumTreeDepth(5).build();
		SamplingOptions twenty = SamplingOptions.builder().warmupIterations(30)
				.sampleIterations(20).maximumTreeDepth(5).build();
		NoUTurnSampler sampler = new NoUTurnSampler();
		ChainResult uninterrupted = sampler.sample(STANDARD_NORMAL, new double[] {0.2, -0.3},
				forty, new MersenneTwister(90210));
		ChainResult first = sampler.sample(STANDARD_NORMAL, new double[] {0.2, -0.3},
				twenty, new MersenneTwister(90210));
		assertNotNull(first.checkpoint().samplerCheckpoint());
		ChainResult resumed = Chains.resume(sampler, STANDARD_NORMAL, first.checkpoint(),
				twenty);
		for (int i = 0; i < 20; i++)
			assertArrayEquals(uninterrupted.sample(i + 20), resumed.sample(i), 0.0);
	}

	@Test public void gradientAndEllipticalSamplersProduceFiniteDraws() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(50)
				.sampleIterations(100).stepSize(0.4).build();
		ChainResult mala = new MetropolisAdjustedLangevin().sample(STANDARD_NORMAL,
				new double[] {0.0, 0.0}, options, new MersenneTwister(11));
		ChainResult barker = new BarkerGradientSampler().sample(STANDARD_NORMAL,
				new double[] {0.0, 0.0}, options, new MersenneTwister(12));
		GaussianReference reference = GaussianReference.diagonal(new double[] {0.0, 0.0},
				new double[] {1.0, 1.0});
		ChainResult elliptical = new EllipticalSliceSampler(reference).sample(
				state -> 0.0, new double[] {0.0, 0.0}, options, new MersenneTwister(13));
		assertEquals(100, mala.size()); assertEquals(100, barker.size());
		assertEquals(100, elliptical.size());
		assertTrue(Double.isFinite(mala.logDensityAt(99)));
	}

	@Test public void adjustedMicrocanonicalSamplerIsExactAndFinite() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(100)
				.sampleIterations(200).stepSize(0.2).leapfrogSteps(5).build();
		ChainResult result = new AdjustedMicrocanonicalLangevin().sample(STANDARD_NORMAL,
				new double[] {0.1, -0.1}, options, new MersenneTwister(44));
		assertEquals(ChainResult.Status.SUCCESS, result.status());
		assertEquals(200, result.size());
		assertTrue(Double.isFinite(result.logDensityAt(199)));
		double[] summary = meanVariance(result, 0);
		assertEquals(0.0, summary[0], 0.4);
		assertEquals(1.0, summary[1], 0.6);
	}

	@Test public void optimizationPathfinderTemperingAndMetricsAreAvailable() {
		OptimizationResult mode = LbfgsOptimizer.maximize(STANDARD_NORMAL,
				new double[] {3.0, -2.0}, 100, 5, 1e-8);
		assertTrue(mode.converged()); assertArrayEquals(new double[] {0.0, 0.0}, mode.point(), 1e-7);
		PathfinderResult path = PathfinderInitializer.initialize(STANDARD_NORMAL,
				new double[] {2.0, 1.0}, new MersenneTwister(2));
		assertEquals(2, path.initialState().length);
		assertEquals(MetricConfiguration.Type.LOW_RANK_DIAGONAL,
				MetricConfiguration.lowRankDiagonal(1).type());
		TemperedLogDensity split = new TemperedLogDensity() {
			@Override public double baseLogDensity(double[] state) { return -0.25 * state[0] * state[0]; }
			@Override public double temperedLogDensity(double[] state) { return -0.25 * state[0] * state[0]; }
		};
		ParallelTemperingResult tempered = ParallelTempering.sample(split, new double[] {0.0},
				new double[] {1.0, 0.5, 0.1}, SamplingOptions.builder()
						.warmupIterations(10).sampleIterations(20).build(), new MersenneTwister(3));
		assertEquals(20, tempered.coldChain().size());
		assertTrue(tempered.attemptedSwaps()[0] > 0);
	}

	@Test public void mcseAndEfficiencyUtilitiesHandleQuantitiesBeyondMeans() {
		double[] draws = new double[1000];
		for (int i = 0; i < draws.length; i++) draws[i] = Math.sin(i * 0.7);
		assertTrue(MonteCarloError.standardDeviationMcse(draws) > 0.0);
		assertTrue(MonteCarloError.quantileMcse(draws, 0.9) > 0.0);
		assertEquals(0.2, MonteCarloError.essPerEvaluation(20.0, 100), 0.0);
		assertEquals(10.0, MonteCarloError.essPerSecond(20.0, 2000000000L), 1e-12);
	}

	@Test public void streamingProgressMetricsCountingAndFitFacadeCompose() {
		final int[] streamed = {0};
		WarmupTrace trace = new WarmupTrace(20, WarmupSchedule.stanDefault());
		SamplingOptions streaming = SamplingOptions.builder().warmupIterations(20)
				.sampleIterations(12).maximumTreeDepth(4)
				.metric(MetricConfiguration.supplied(new double[][] {{1.0, 0.2}, {0.2, 1.0}}))
				.progressListener(trace).storeDraws(false)
				.drawSink((index, state, logDensity, statistics) -> streamed[0]++).build();
		EvaluationCounter counted = new EvaluationCounter(STANDARD_NORMAL);
		ChainResult result = new NoUTurnSampler().sample(counted, new double[] {0.0, 0.0},
				streaming, new MersenneTwister(30));
		assertEquals(0, result.size()); assertEquals(12, streamed[0]);
		assertTrue(counted.gradientEvaluations() > 12);
		assertEquals(20, trace.entries().size());

		SamplingOptions fitOptions = SamplingOptions.builder().warmupIterations(20)
				.sampleIterations(12).maximumTreeDepth(4)
				.metric(MetricConfiguration.blockDiagonal(new int[] {0}, new int[] {1})).build();
		Fit fit = Inference.fit(new NoUTurnSampler(), STANDARD_NORMAL, "normal-v1",
				new double[][] {{-0.2, 0.1}, {0.2, -0.1}}, fitOptions, 91L, 2);
		assertEquals(2, fit.chains().length);
		assertEquals(91L, fit.manifest().seed());
		assertEquals(64, fit.manifest().modelHash().length());
	}

	private static double[] meanVariance(ChainResult result, int coordinate) {
		double mean = 0.0;
		for (int i = 0; i < result.size(); i++) mean += result.valueAt(i, coordinate);
		mean /= result.size();
		double variance = 0.0;
		for (int i = 0; i < result.size(); i++) {
			double difference = result.valueAt(i, coordinate) - mean;
			variance += difference * difference;
		}
		return new double[] {mean, variance / (result.size() - 1.0)};
	}
}
