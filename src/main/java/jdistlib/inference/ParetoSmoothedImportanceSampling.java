/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Pareto-tail smoothing for log importance ratios, with a diagnostic shape estimate. */
public final class ParetoSmoothedImportanceSampling {
	private ParetoSmoothedImportanceSampling() {}
	public static Result smooth(double[] logRatios) {
		if (logRatios == null || logRatios.length < 5) throw new IllegalArgumentException("at least five ratios required");
		double maximum = Double.NEGATIVE_INFINITY;
		for (double value : logRatios) { if (!Double.isFinite(value)) throw new IllegalArgumentException("ratios must be finite"); maximum = Math.max(maximum, value); }
		double[] weights = new double[logRatios.length]; Integer[] order = new Integer[weights.length];
		for (int i = 0; i < weights.length; i++) { weights[i] = Math.exp(logRatios[i] - maximum); order[i] = i; }
		Arrays.sort(order, (a, b) -> Double.compare(weights[a], weights[b]));
		int tailLength = Math.max(3, Math.min(weights.length / 5, (int) Math.ceil(3.0 * Math.sqrt(weights.length))));
		int thresholdIndex = weights.length - tailLength - 1;
		double threshold = weights[order[Math.max(0, thresholdIndex)]];
		double mean = 0.0, variance = 0.0;
		for (int i = weights.length - tailLength; i < weights.length; i++) mean += weights[order[i]] - threshold;
		mean /= tailLength;
		for (int i = weights.length - tailLength; i < weights.length; i++) { double d = weights[order[i]] - threshold - mean; variance += d * d; }
		variance /= Math.max(1.0, tailLength - 1.0);
		double k = variance > 0.0 ? 0.5 * (1.0 - mean * mean / variance) : 0.0;
		k = Math.max(-0.5, Math.min(5.0, k));
		double sigma = Math.max(1e-15, mean * (1.0 - k));
		double largest = weights[order[weights.length - 1]];
		for (int rank = 0; rank < tailLength; rank++) {
			double probability = (rank + 0.5) / tailLength;
			double excess = Math.abs(k) < 1e-8 ? -sigma * Math.log1p(-probability)
					: sigma / k * (Math.pow(1.0 - probability, -k) - 1.0);
			weights[order[weights.length - tailLength + rank]] = Math.min(largest, threshold + excess);
		}
		double sum = 0.0; for (double weight : weights) sum += weight;
		double[] smoothedLog = new double[weights.length];
		for (int i = 0; i < weights.length; i++) smoothedLog[i] = Math.log(weights[i] / sum);
		return new Result(smoothedLog, k);
	}
	public static final class Result {
		private final double[] logWeights; private final double paretoK;
		private Result(double[] logWeights, double paretoK) { this.logWeights = logWeights; this.paretoK = paretoK; }
		public double[] logWeights() { return logWeights.clone(); }
		public double paretoK() { return paretoK; }
		public boolean reliable() { return paretoK <= 0.7; }
	}
}
