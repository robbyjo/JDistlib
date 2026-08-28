/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

/** Shared exact sequential-conditional dynamic program for count vectors. */
final class DiscreteMultivariateProbability {
	private DiscreteMultivariateProbability() {}

	interface ConditionalMass {
		double mass(int category, int count, int remaining);
	}

	static double probability(int[] lower, int[] upper, int total,
			int categories, ConditionalMass conditionalMass) {
		if (lower == null || upper == null || conditionalMass == null ||
				categories < 1 || lower.length != categories ||
				upper.length != categories || total < 0) return Double.NaN;
		int[] lo = new int[categories];
		int[] hi = new int[categories];
		for (int i = 0; i < categories; i++) {
			lo[i] = Math.max(0, lower[i]);
			hi[i] = Math.min(total, upper[i]);
			if (lo[i] > hi[i]) return 0.0;
		}
		double[] current = new double[total + 1];
		current[total] = 1.0;
		for (int category = 0; category + 1 < categories; category++) {
			double[] next = new double[total + 1];
			double[] compensation = new double[total + 1];
			for (int remaining = 0; remaining <= total; remaining++) {
				if (!(current[remaining] > 0.0)) continue;
				int maximum = Math.min(hi[category], remaining);
				for (int count = lo[category]; count <= maximum; count++) {
					double mass = conditionalMass.mass(category, count, remaining);
					if (!(mass > 0.0)) continue;
					int destination = remaining - count;
					double term = current[remaining] * mass;
					double corrected = term - compensation[destination];
					double sum = next[destination] + corrected;
					compensation[destination] = (sum - next[destination]) - corrected;
					next[destination] = sum;
				}
			}
			current = next;
		}
		double result = 0.0;
		for (int remaining = lo[categories - 1];
				remaining <= hi[categories - 1] && remaining <= total; remaining++)
			result += current[remaining];
		return Math.max(0.0, Math.min(1.0, result));
	}

	static double logBetaBinomialMass(int count, int size, double alpha,
			double beta) {
		if (count < 0 || count > size) return Double.NEGATIVE_INFINITY;
		return lgammafn(size + 1.0) - lgammafn(count + 1.0) -
				lgammafn(size - count + 1.0) + lgammafn(count + alpha) +
				lgammafn(size - count + beta) - lgammafn(size + alpha + beta) +
				lgammafn(alpha + beta) - lgammafn(alpha) - lgammafn(beta);
	}
}
