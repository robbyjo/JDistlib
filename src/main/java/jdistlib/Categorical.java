/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Finite categorical distribution over numeric outcomes. */
public class Categorical extends GenericDistribution {
	private static double[][] normalize(double[] outcomes, double[] weights) {
		if (outcomes == null || weights == null || outcomes.length == 0
				|| outcomes.length != weights.length) return null;
		double[][] pairs = new double[outcomes.length][2];
		double sum = 0.0;
		for (int i = 0; i < outcomes.length; i++) {
			if (!Double.isFinite(outcomes[i]) || !(weights[i] >= 0.0)
					|| Double.isInfinite(weights[i])) return null;
			pairs[i][0] = outcomes[i];
			pairs[i][1] = weights[i];
			sum += weights[i];
		}
		if (!(sum > 0.0) || Double.isInfinite(sum)) return null;
		Arrays.sort(pairs, (a, b) -> Double.compare(a[0], b[0]));
		for (double[] pair : pairs) pair[1] /= sum;
		return pairs;
	}

	public static double density(double x, double[] outcomes, double[] weights,
			boolean giveLog) {
		double[][] pairs = normalize(outcomes, weights);
		if (Double.isNaN(x) || pairs == null) return Double.NaN;
		double probability = 0.0;
		for (double[] pair : pairs) if (pair[0] == x) probability += pair[1];
		return giveLog ? Math.log(probability) : probability;
	}

	public static double cumulative(double x, double[] outcomes, double[] weights,
			boolean lowerTail, boolean logP) {
		double[][] pairs = normalize(outcomes, weights);
		if (Double.isNaN(x) || pairs == null) return Double.NaN;
		double probability = 0.0;
		for (double[] pair : pairs) if (pair[0] <= x) probability += pair[1];
		if (!lowerTail) probability = 1.0 - probability;
		return logP ? Math.log(probability) : probability;
	}

	public static double quantile(double p, double[] outcomes, double[] weights,
			boolean lowerTail, boolean logP) {
		double[][] pairs = normalize(outcomes, weights);
		if (pairs == null || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		if (!lowerTail) probability = logP ? -Math.expm1(p) : 1.0 - p;
		double cumulative = 0.0;
		for (double[] pair : pairs) {
			cumulative += pair[1];
			if (cumulative >= probability) return pair[0];
		}
		return pairs[pairs.length - 1][0];
	}

	public static double random(double[] outcomes, double[] weights,
			RandomEngine random) {
		return quantile(random.nextDouble(), outcomes, weights, true, false);
	}

	public static double[] random(int n, double[] outcomes, double[] weights,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(outcomes, weights, random);
		return result;
	}

	private final double[] outcomes;
	private final double[] weights;

	public Categorical(double[] outcomes, double[] weights) {
		this.outcomes = outcomes.clone();
		this.weights = weights.clone();
	}

	public Categorical(double[] outcomes) {
		this.outcomes = outcomes.clone();
		this.weights = new double[outcomes.length];
		Arrays.fill(this.weights, 1.0);
	}

	@Override public double density(double x, boolean log) {
		return density(x, outcomes, weights, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, outcomes, weights, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, outcomes, weights, lowerTail, logP);
	}
	@Override public double random() { return random(outcomes, weights, random); }
}
