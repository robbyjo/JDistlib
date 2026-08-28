/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Chunked deterministic continuation based on MCSE, never on R-hat alone. */
public final class PrecisionContinuation {
	private PrecisionContinuation() {}
	public static PrecisionContinuationResult run(Sampler sampler, LogDensity target, double[] initial,
			SamplingOptions baseOptions, PrecisionGoal goal, int chunkDraws, RandomEngine random) {
		if (sampler == null || target == null || initial == null || baseOptions == null || goal == null || chunkDraws < 4 || random == null)
			throw new IllegalArgumentException("sampler, target, options, goal, chunk size, and random are required");
		if (goal.coordinate() >= initial.length) throw new IllegalArgumentException("precision coordinate out of range");
		ChainResult combined = null; double mcse = Double.POSITIVE_INFINITY; boolean met = false; int chunks = 0;
		for (; chunks < goal.maximumChunks(); chunks++) {
			int continuationWarmup = chunks == 0 || sampler instanceof ResumableSampler
					? baseOptions.warmupIterations() : 0;
			SamplingOptions options = baseOptions.toBuilder().sampleIterations(chunkDraws).thinning(1)
					.warmupIterations(continuationWarmup).storeDraws(true).build();
			ChainResult next = chunks == 0 ? sampler.sample(target, initial, options, random)
					: Chains.resume(sampler, target, combined.checkpoint(), options);
			combined = append(combined, next); double[] values = goal.evaluate(combined);
			if (values.length >= 4) mcse = MonteCarloError.meanMcse(values);
			if (combined.size() >= goal.minimumDraws() && healthy(combined) && meets(goal, values, mcse)) { met = true; chunks++; break; }
			if (next.status() != ChainResult.Status.SUCCESS) { chunks++; break; }
		}
		return new PrecisionContinuationResult(combined, chunks, mcse, met);
	}
	private static boolean meets(PrecisionGoal goal, double[] values, double mcse) { if (!Double.isFinite(mcse)) return false;
		if (!Double.isNaN(goal.absoluteMcse()) && mcse > goal.absoluteMcse()) return false; double mean = 0.0; for (double value : values) mean += value; mean /= values.length;
		return Double.isNaN(goal.relativeMcse()) || mcse <= goal.relativeMcse() * Math.max(Math.abs(mean), 1e-12); }
	private static boolean healthy(ChainResult chain) { if (chain.status() != ChainResult.Status.SUCCESS) return false;
		for (int i = 0; i < chain.size(); i++) if (chain.statisticsAt(i).divergent()) return false; return true; }
	private static ChainResult append(ChainResult first, ChainResult second) { if (first == null) return second; int count = first.size() + second.size(); double[][] samples = new double[count][];
		double[] densities = new double[count]; IterationStats[] stats = new IterationStats[count]; int index = 0;
		for (int i = 0; i < first.size(); i++, index++) { samples[index] = first.sample(i); densities[index] = first.logDensityAt(i); stats[index] = first.statisticsAt(i); }
		for (int i = 0; i < second.size(); i++, index++) { samples[index] = second.sample(i); densities[index] = second.logDensityAt(i); stats[index] = second.statisticsAt(i); }
		List<String> warnings = new ArrayList<String>(first.warnings()); warnings.addAll(second.warnings()); return new ChainResult(samples, densities, stats,
				first.warmup(), second.checkpoint(), second.status(), warnings); }
}
