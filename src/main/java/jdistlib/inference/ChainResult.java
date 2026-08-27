/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable retained samples, sampler statistics, adaptation and restart state. */
public final class ChainResult {
	public enum Status { SUCCESS, CANCELLED, INVALID_INITIAL_STATE, NUMERICAL_FAILURE }
	private final double[][] samples;
	private final double[] logDensities;
	private final IterationStats[] statistics;
	private final WarmupResult warmup;
	private final ChainCheckpoint checkpoint;
	private final Status status;
	private final List<String> warnings;

	public ChainResult(double[][] samples, double[] logDensities,
			IterationStats[] statistics, WarmupResult warmup,
			ChainCheckpoint checkpoint, Status status, List<String> warnings) {
		if (samples == null || logDensities == null || statistics == null
				|| samples.length != logDensities.length
				|| samples.length != statistics.length || status == null) {
			throw new IllegalArgumentException("chain arrays and status must match");
		}
		this.samples = new double[samples.length][];
		for (int i = 0; i < samples.length; i++) this.samples[i] = samples[i].clone();
		this.logDensities = logDensities.clone();
		this.statistics = statistics.clone();
		this.warmup = warmup;
		this.checkpoint = checkpoint;
		this.status = status;
		this.warnings = Collections.unmodifiableList(new ArrayList<String>(warnings));
	}
	public int size() { return samples.length; }
	public int dimension() { return samples.length == 0 ? 0 : samples[0].length; }
	public double[][] samples() {
		double[][] copy = new double[samples.length][];
		for (int i = 0; i < samples.length; i++) copy[i] = samples[i].clone();
		return copy;
	}
	public double[] sample(int index) { return samples[index].clone(); }
	/** Returns one retained unconstrained value without copying the chain. */
	public double valueAt(int draw, int coordinate) { return samples[draw][coordinate]; }
	/** Returns one retained log density without copying the chain. */
	public double logDensityAt(int draw) { return logDensities[draw]; }
	/** Returns immutable statistics for one retained transition. */
	public IterationStats statisticsAt(int draw) { return statistics[draw]; }
	public double[] logDensities() { return logDensities.clone(); }
	public IterationStats[] statistics() { return statistics.clone(); }
	public WarmupResult warmup() { return warmup; }
	public ChainCheckpoint checkpoint() { return checkpoint; }
	public Status status() { return status; }
	public List<String> warnings() { return warnings; }
	public double[][] constrainedSamples(BayesianModel model) {
		if (model == null || model.dimension() != dimension())
			throw new IllegalArgumentException("model dimension does not match chain");
		double[][] result = new double[samples.length][];
		for (int i = 0; i < samples.length; i++) result[i] = model.constrain(samples[i]);
		return result;
	}
}
