/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Pareto-tail smoothing for log importance ratios, with a diagnostic shape estimate. */
public final class ParetoSmoothedImportanceSampling {
	private ParetoSmoothedImportanceSampling() {}
	/** Uses the empirical-Bayes generalized Pareto fit used by posterior/loo,
	 * with independent-draw relative efficiency (r_eff=1). */
	public static Result smooth(double[] logRatios) {
		if (logRatios == null || logRatios.length < 5) throw new IllegalArgumentException("at least five ratios required");
		double maximum = Double.NEGATIVE_INFINITY;
		for (double value : logRatios) { if (!Double.isFinite(value)) throw new IllegalArgumentException("ratios must be finite"); maximum = Math.max(maximum, value); }
		double[] logs = new double[logRatios.length]; Integer[] order = new Integer[logs.length];
		for (int i = 0; i < logs.length; i++) { logs[i] = logRatios[i] - maximum; order[i] = i; }
		Arrays.sort(order, (a, b) -> Double.compare(logs[a], logs[b]));
		int tailLength = (int) Math.ceil(Math.min(logs.length * .2, 3 * Math.sqrt(logs.length)));
		double k = Double.POSITIVE_INFINITY;
		if (tailLength >= 5) {
			int first = logs.length - tailLength;
			double threshold = Math.exp(logs[order[first - 1]]);
			double[] excess = new double[tailLength];
			for (int i = 0; i < tailLength; i++) excess[i] = Math.exp(logs[order[first + i]]) - threshold;
			double[] fit = fitTail(excess); k = fit[0];
			if (Double.isFinite(k)) for (int i = 0; i < tailLength; i++) {
				double logSurvival = Math.log1p(-(i + .5) / tailLength);
				double quantile = k == 0 ? -fit[1] * logSurvival : fit[1] * Math.expm1(-k * logSurvival) / k;
				logs[order[first + i]] = Math.min(0, Math.log(threshold + quantile));
			}
		}
		double normalizer = PredictiveMath.logSumExp(logs);
		for (int i = 0; i < logs.length; i++) logs[i] -= normalizer;
		return new Result(logs, k);
	}

	// Empirical-Bayes inverse-scale grid with weak shape prior (Zhang/Stephens).
	private static double[] fitTail(double[] excess) {
		int n = excess.length, grid = 30 + (int) Math.sqrt(n);
		double quarter = excess[(int) Math.floor(n / 4.0 + .5) - 1];
		if (!(quarter > excess[0])) return new double[] {Double.POSITIVE_INFINITY, Double.NaN};
		double[] theta = new double[grid], logWeight = new double[grid];
		for (int j = 0; j < grid; j++) {
			theta[j] = 1 / excess[n - 1] + (1 - Math.sqrt(grid / (j + .5))) / 3 / quarter;
			double shape = 0;
			for (double value : excess) shape += Math.log1p(-theta[j] * value) / n;
			logWeight[j] = n * (Math.log(-theta[j] / shape) - shape - 1);
		}
		double normalizer = PredictiveMath.logSumExp(logWeight), thetaMean = 0;
		for (int j = 0; j < grid; j++) thetaMean += theta[j] * Math.exp(logWeight[j] - normalizer);
		double shape = 0;
		for (double value : excess) shape += Math.log1p(-thetaMean * value) / n;
		double sigma = -shape / thetaMean;
		shape = (shape * n + 5) / (n + 10);
		return Double.isFinite(shape) && sigma > 0 ? new double[] {shape, sigma}
				: new double[] {Double.POSITIVE_INFINITY, Double.NaN};
	}
	public static final class Result {
		private final double[] logWeights; private final double paretoK;
		private Result(double[] logWeights, double paretoK) { this.logWeights = logWeights; this.paretoK = paretoK; }
		public double[] logWeights() { return logWeights.clone(); }
		public double paretoK() { return paretoK; }
		public boolean reliable() { return paretoK <= 0.7; }
	}
}
