/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Package-private retained-sample builder shared by samplers. */
final class ChainAccumulator {
	private final List<double[]> samples = new ArrayList<double[]>();
	private final List<Double> logDensities = new ArrayList<Double>();
	private final List<IterationStats> statistics = new ArrayList<IterationStats>();
	private final List<String> warnings = new ArrayList<String>();

	void add(double[] state, double logDensity, IterationStats stats) {
		samples.add(state.clone()); logDensities.add(logDensity); statistics.add(stats);
	}
	void warn(String warning) { warnings.add(warning); }
	ChainResult result(double[] state, double logDensity, int completed,
			RandomEngine random, WarmupResult warmup, ChainResult.Status status) {
		double[][] sampleArray = samples.toArray(new double[samples.size()][]);
		double[] densityArray = new double[logDensities.size()];
		for (int i = 0; i < densityArray.length; i++) densityArray[i] = logDensities.get(i);
		IterationStats[] statsArray = statistics.toArray(new IterationStats[statistics.size()]);
		return new ChainResult(sampleArray, densityArray, statsArray, warmup,
				new ChainCheckpoint(state, logDensity, completed, random), status, warnings);
	}
}
