/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Multinomial mass and random generation for a vector of category counts. */
public final class Multinomial {
	private Multinomial() {}

	private static double[] probabilities(double[] weights) {
		if (weights == null || weights.length == 0) return null;
		double sum = 0.0;
		for (double weight : weights) {
			if (!(weight >= 0.0) || Double.isInfinite(weight)) return null;
			sum += weight;
		}
		if (!(sum > 0.0) || Double.isInfinite(sum)) return null;
		double[] result = weights.clone();
		for (int i = 0; i < result.length; i++) result[i] /= sum;
		return result;
	}

	public static double density(double[] x, int size, double[] weights,
			boolean giveLog) {
		double[] p = probabilities(weights);
		if (x == null || p == null || x.length != p.length || size < 0) {
			return Double.NaN;
		}
		double total = 0.0;
		double result = lgammafn(size + 1.0);
		for (int i = 0; i < x.length; i++) {
			if (x[i] < 0.0 || x[i] != Math.rint(x[i]) || Double.isInfinite(x[i])) {
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			}
			total += x[i];
			if (p[i] == 0.0 && x[i] > 0.0) {
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			}
			result -= lgammafn(x[i] + 1.0);
			if (x[i] > 0.0) result += x[i] * Math.log(p[i]);
		}
		if (total != size) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		return giveLog ? result : Math.exp(result);
	}

	/** Exact inclusive rectangle probability for the category counts. */
	public static double probability(int[] lower, int[] upper, int size,
			double[] weights) {
		final double[] p = probabilities(weights);
		if (p == null || size < 0) return Double.NaN;
		final double[] remainingProbability = new double[p.length];
		double sum = 0.0;
		for (int i = p.length - 1; i >= 0; i--) {
			sum += p[i];
			remainingProbability[i] = sum;
		}
		return DiscreteMultivariateProbability.probability(lower, upper, size,
				p.length, (category, count, remaining) -> {
				double conditional = remainingProbability[category] == 0.0 ? 0.0 :
						p[category] / remainingProbability[category];
				return Binomial.density(count, remaining, conditional, false);
			});
	}

	/** Exact lower-orthant probability {@code P(X[i] <= upper[i], all i)}. */
	public static double cumulative(int[] upper, int size, double[] weights) {
		return upper == null ? Double.NaN : probability(new int[upper.length],
				upper, size, weights);
	}

	public static int[] random(int size, double[] weights, RandomEngine random) {
		return random(size, weights, random, Binomial.create_random_state());
	}

	/**
	 * Draws a multinomial vector using sequential conditional binomials and
	 * Kahan compensated probability arithmetic, matching current R's
	 * {@code rmultinom}. The supplied binomial state also selects corrected or
	 * legacy BTPE behavior.
	 */
	public static int[] random(int size, double[] weights, RandomEngine random,
			Binomial.RandomState binomialState) {
		double[] p = probabilities(weights);
		if (p == null || size < 0 || random == null) return null;
		if (binomialState == null) binomialState = Binomial.create_random_state();
		int[] result = new int[p.length];
		double remainingProbability = 0.0;
		double compensation = 0.0;
		for (double probability : p) {
			double corrected = probability - compensation;
			double next = remainingProbability + corrected;
			compensation = (next - remainingProbability) - corrected;
			remainingProbability = next;
		}
		int remainingSize = size;
		for (int category = 0; category < p.length - 1; category++) {
			if (p[category] != 0.0) {
				double conditional = p[category] / remainingProbability;
				result[category] = conditional < 1.0
						? (int) Binomial.random(remainingSize, conditional, random,
								binomialState)
						: remainingSize;
				remainingSize -= result[category];
			}
			if (remainingSize <= 0) return result;
			double corrected = -p[category] - compensation;
			double next = remainingProbability + corrected;
			compensation = (next - remainingProbability) - corrected;
			remainingProbability = next;
		}
		result[p.length - 1] = remainingSize;
		return result;
	}

	public static int[][] random(int n, int size, double[] weights,
			RandomEngine random) {
		return random(n, size, weights, random, Binomial.create_random_state());
	}

	public static int[][] random(int n, int size, double[] weights,
			RandomEngine random, Binomial.RandomState binomialState) {
		if (n < 0) return null;
		int[][] result = new int[n][];
		for (int i = 0; i < n; i++) {
			result[i] = random(size, weights, random, binomialState);
		}
		return result;
	}
}
