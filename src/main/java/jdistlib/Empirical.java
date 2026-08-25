/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Discrete empirical distribution that samples observations with replacement. */
public class Empirical extends GenericDistribution {
	private final double[] sample;

	public Empirical(double[] sample) {
		if (sample == null || sample.length < 2) {
			throw new IllegalArgumentException("at least two observations are required");
		}
		this.sample = sample.clone();
		for (double value : this.sample) {
			if (!Double.isFinite(value)) {
				throw new IllegalArgumentException("observations must be finite");
			}
		}
		Arrays.sort(this.sample);
	}

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		int count = 0;
		for (double value : sample) if (Math.abs(value - x) < 1.4901161193847656e-8) count++;
		double result = count / (double) sample.length;
		return log ? Math.log(result) : result;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		int low = 0;
		int high = sample.length;
		while (low < high) {
			int middle = (low + high) >>> 1;
			if (sample[middle] <= x) low = middle + 1;
			else high = middle;
		}
		double result = low / (double) sample.length;
		if (!lowerTail) result = 1.0 - result;
		return logP ? Math.log(result) : result;
	}

	/** Uses the inverse empirical CDF (R quantile type 1). */
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double probability = logP ? Math.exp(p) : p;
		if (!lowerTail) probability = logP ? -Math.expm1(p) : 1.0 - p;
		if (probability <= 0.0) return sample[0];
		int index = (int) Math.ceil(probability * sample.length) - 1;
		return sample[Math.min(index, sample.length - 1)];
	}

	public static double random(double[] sample, RandomEngine random) {
		if (sample == null || sample.length == 0) return Double.NaN;
		int index = (int) Math.floor(random.nextDouble() * sample.length);
		return sample[Math.min(index, sample.length - 1)];
	}

	public static double[] random(int n, double[] sample, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(sample, random);
		return result;
	}

	@Override public double random() { return random(sample, random); }
}
