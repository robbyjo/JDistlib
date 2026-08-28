/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** MCSE and efficiency helpers for one stationary retained sequence. */
public final class MonteCarloError {
	private MonteCarloError() {}
	public static double effectiveSampleSize(double[] values) {
		if (values == null || values.length < 4) throw new IllegalArgumentException("at least four draws are required");
		double mean = mean(values), variance = 0.0;
		for (double value : values) variance += square(value - mean);
		if (variance == 0.0) return values.length;
		double sum = 0.0, previous = Double.POSITIVE_INFINITY;
		for (int lag = 1; lag + 1 < values.length; lag += 2) {
			double pair = autocorrelation(values, mean, variance, lag)
					+ autocorrelation(values, mean, variance, lag + 1);
			if (pair < 0.0) break;
			pair = Math.min(pair, previous); previous = pair; sum += pair;
		}
		return Math.max(1.0, Math.min(values.length, values.length / (1.0 + 2.0 * sum)));
	}
	public static double standardDeviationMcse(double[] values) {
		double mean = mean(values), variance = 0.0;
		double[] squared = new double[values.length];
		for (int i = 0; i < values.length; i++) { squared[i] = square(values[i] - mean); variance += squared[i]; }
		variance /= Math.max(1.0, values.length - 1.0);
		return Math.sqrt(variance / (2.0 * effectiveSampleSize(squared)));
	}
	public static double quantileMcse(double[] values, double probability) {
		if (!(probability > 0.0 && probability < 1.0)) throw new IllegalArgumentException("probability must be between zero and one");
		double quantile = quantile(values, probability);
		double[] indicator = new double[values.length];
		for (int i = 0; i < values.length; i++) indicator[i] = values[i] <= quantile ? 1.0 : 0.0;
		double probabilitySe = Math.sqrt(probability * (1.0 - probability)
				/ effectiveSampleSize(indicator));
		double lower = quantile(values, Math.max(0.0, probability - probabilitySe));
		double upper = quantile(values, Math.min(1.0, probability + probabilitySe));
		return 0.5 * (upper - lower);
	}
	public static double essPerEvaluation(double ess, long evaluations) {
		return evaluations <= 0L ? Double.NaN : ess / evaluations;
	}
	public static double essPerSecond(double ess, long elapsedNanoseconds) {
		return elapsedNanoseconds <= 0L ? Double.NaN : ess * 1e9 / elapsedNanoseconds;
	}
	private static double autocorrelation(double[] x, double mean, double variance, int lag) {
		double covariance = 0.0;
		for (int i = 0; i + lag < x.length; i++) covariance += (x[i] - mean) * (x[i + lag] - mean);
		return covariance / variance;
	}
	private static double mean(double[] x) { double result = 0.0; for (double v : x) result += v; return result / x.length; }
	private static double square(double x) { return x * x; }
	private static double quantile(double[] x, double p) {
		double[] sorted = x.clone(); Arrays.sort(sorted); double index = p * (sorted.length - 1.0);
		int lower = (int) Math.floor(index), upper = Math.min(sorted.length - 1, lower + 1);
		return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
	}
}
