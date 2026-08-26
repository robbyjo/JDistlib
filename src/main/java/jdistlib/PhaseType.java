/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Continuous phase-type law with an optional atom at zero.
 *
 * <p>The initial transient-state probabilities need only sum to at most one;
 * the remainder is the probability of starting in the absorbing state. The
 * rate matrix is a transient subgenerator.</p>
 */
public final class PhaseType extends GenericDistribution
		implements SupportedDistribution, AtomAwareDistribution {
	private final double[] initial;
	private final double[][] rates;
	private final double[] exitRates;
	private final double atom;

	public PhaseType(double[] initial, double[][] rates) {
		validate(initial, rates);
		this.initial = initial.clone();
		this.rates = copy(rates);
		exitRates = exits(rates);
		double sum = 0.0;
		for (double value : initial) sum += value;
		atom = Math.max(0.0, 1.0 - sum);
	}

	private static void validate(double[] initial, double[][] rates) {
		if (initial == null || rates == null || initial.length == 0
				|| rates.length != initial.length) {
			throw new IllegalArgumentException("matching nonempty initial vector and rate matrix are required");
		}
		double sum = 0.0;
		for (int i = 0; i < initial.length; i++) {
			if (!(initial[i] >= 0.0) || !Double.isFinite(initial[i])
					|| rates[i] == null || rates[i].length != initial.length) {
				throw new IllegalArgumentException("invalid initial probabilities or square rate matrix");
			}
			sum += initial[i];
			double row = 0.0;
			for (int j = 0; j < initial.length; j++) {
				double value = rates[i][j];
				if (!Double.isFinite(value) || (i == j ? value >= 0.0 : value < 0.0)) {
					throw new IllegalArgumentException("rates must be a finite subgenerator");
				}
				row += value;
			}
			if (row > 1e-12 * Math.max(1.0, -rates[i][i])) {
				throw new IllegalArgumentException("rate-matrix row sums must be nonpositive");
			}
		}
		if (sum > 1.0 + 1e-12) {
			throw new IllegalArgumentException("initial probabilities must sum to at most one");
		}
		boolean[] canExit = new boolean[initial.length];
		for (int i = 0; i < initial.length; i++) {
			double row = 0.0;
			for (double value : rates[i]) row += value;
			canExit[i] = row < -1e-14 * Math.max(1.0, -rates[i][i]);
		}
		for (int pass = 0; pass < initial.length; pass++) {
			for (int i = 0; i < initial.length; i++) {
				for (int j = 0; !canExit[i] && j < initial.length; j++) {
					if (rates[i][j] > 0.0 && canExit[j]) canExit[i] = true;
				}
			}
		}
		for (int i = 0; i < initial.length; i++) {
			if (initial[i] > 0.0 && !canExit[i]) {
				throw new IllegalArgumentException("every initially reachable class must lead to absorption");
			}
		}
	}

	private static double[][] copy(double[][] matrix) {
		double[][] result = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) result[i] = matrix[i].clone();
		return result;
	}

	private static double[] exits(double[][] matrix) {
		double[] result = new double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			for (double value : matrix[i]) result[i] -= value;
			result[i] = Math.max(0.0, result[i]);
		}
		return result;
	}

	private static double[][] multiply(double[][] a, double[][] b) {
		int n = a.length;
		double[][] result = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < n; k++) {
				double aik = a[i][k];
				if (aik == 0.0) continue;
				for (int j = 0; j < n; j++) result[i][j] += aik * b[k][j];
			}
		}
		return result;
	}

	private static double[][] exponential(double[][] rates, double x) {
		int n = rates.length;
		double norm = 0.0;
		for (int i = 0; i < n; i++) {
			double row = 0.0;
			for (int j = 0; j < n; j++) row += Math.abs(rates[i][j] * x);
			norm = Math.max(norm, row);
		}
		if (Double.isInfinite(norm)) return new double[n][n];
		int squarings = norm <= 0.5 ? 0
				: Math.max(0, (int) Math.ceil(Math.log(norm / 0.5) / Math.log(2.0)));
		double divisor = Math.scalb(1.0, squarings);
		double[][] scaled = new double[n][n];
		double[][] result = new double[n][n];
		double[][] term = new double[n][n];
		for (int i = 0; i < n; i++) {
			result[i][i] = 1.0;
			term[i][i] = 1.0;
			for (int j = 0; j < n; j++) scaled[i][j] = rates[i][j] * x / divisor;
		}
		for (int order = 1; order <= 100; order++) {
			term = multiply(term, scaled);
			double largest = 0.0;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					term[i][j] /= order;
					result[i][j] += term[i][j];
					largest = Math.max(largest, Math.abs(term[i][j]));
				}
			}
			if (largest < 2e-16) break;
		}
		for (int i = 0; i < squarings; i++) result = multiply(result, result);
		return result;
	}

	private static double product(double[] left, double[][] matrix, double[] right) {
		double result = 0.0;
		for (int i = 0; i < left.length; i++) {
			for (int j = 0; j < right.length; j++) {
				result += left[i] * matrix[i][j] * right[j];
			}
		}
		return result;
	}

	public static double density(double x, double[] initial, double[][] rates,
			boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		validate(initial, rates);
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double probability;
		if (x == 0.0) {
			double sum = 0.0;
			for (double value : initial) sum += value;
			probability = Math.max(0.0, 1.0 - sum);
		} else {
			probability = product(initial, exponential(rates, x), exits(rates));
		}
		return log ? Math.log(probability) : probability;
	}

	public static double cumulative(double x, double[] initial, double[][] rates,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		validate(initial, rates);
		if (x < 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double[] ones = new double[initial.length];
		java.util.Arrays.fill(ones, 1.0);
		double survival = product(initial, exponential(rates, x), ones);
		survival = Math.max(0.0, Math.min(1.0, survival));
		double probability = lowerTail ? 1.0 - survival : survival;
		return logP ? Math.log(probability) : probability;
	}

	public static double quantile(double p, double[] initial, double[][] rates,
			boolean lowerTail, boolean logP) {
		validate(initial, rates);
		if (DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double probability = logP ? Math.exp(p) : p;
		double target = lowerTail ? probability : 1.0 - probability;
		if (target <= cumulative(0.0, initial, rates, true, false)) return 0.0;
		if (target >= 1.0) return Double.POSITIVE_INFINITY;
		double low = 0.0;
		double high = 1.0;
		while (cumulative(high, initial, rates, true, false) < target) high *= 2.0;
		for (int i = 0; i < 120; i++) {
			double middle = low + (high - low) * 0.5;
			if (cumulative(middle, initial, rates, true, false) >= target) high = middle;
			else low = middle;
		}
		return (low + high) * 0.5;
	}

	public static double random(double[] initial, double[][] rates,
			RandomEngine random) {
		validate(initial, rates);
		double selector = random.nextDouble();
		double cumulative = 0.0;
		int state = -1;
		for (int i = 0; i < initial.length; i++) {
			cumulative += initial[i];
			if (selector < cumulative) { state = i; break; }
		}
		if (state < 0) return 0.0;
		double value = 0.0;
		while (state >= 0) {
			double totalRate = -rates[state][state];
			value += Exponential.random(1.0 / totalRate, random);
			double transition = random.nextDouble() * totalRate;
			int next = -1;
			for (int j = 0; j < initial.length; j++) {
				if (j == state) continue;
				transition -= rates[state][j];
				if (transition < 0.0) { next = j; break; }
			}
			state = next;
		}
		return value;
	}

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (x == 0.0) return log ? Math.log(atom) : atom;
		double probability = product(initial, exponential(rates, x), exitRates);
		return log ? Math.log(probability) : probability;
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, initial, rates, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, initial, rates, lowerTail, logP);
	}
	@Override public double random() { return random(initial, rates, random); }
	@Override public double atomProbability(double x) { return x == 0.0 ? atom : 0.0; }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
	public double[] getInitialProbabilities() { return initial.clone(); }
	public double[][] getRateMatrix() { return copy(rates); }
}
