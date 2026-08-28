/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import jdistlib.AtomAwareDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Immutable equal-weight empirical distribution used by reproducible fallbacks. */
public final class EmpiricalDistribution extends GenericDistribution
		implements AtomAwareDistribution, SupportedDistribution {
	private final double[] sorted;

	public EmpiricalDistribution(double[] observations) {
		if (observations == null || observations.length == 0)
			throw new IllegalArgumentException("at least one observation is required");
		this.sorted = observations.clone();
		for (double value : sorted) if (!Double.isFinite(value))
			throw new IllegalArgumentException("observations must be finite");
		Arrays.sort(sorted);
	}

	public int size() { return sorted.length; }
	public double[] observations() { return sorted.clone(); }

	@Override public double density(double x, boolean log) {
		double mass = atomProbability(x);
		return log ? Math.log(mass) : mass;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		int index = upperBound(x);
		double value = index / (double) sorted.length;
		if (!lowerTail) value = 1.0 - value;
		return logP ? Math.log(value) : value;
	}

	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		if (logP) probability = Math.exp(probability);
		if (!lowerTail) probability = 1.0 - probability;
		if (probability < 0.0 || probability > 1.0 || Double.isNaN(probability)) return Double.NaN;
		if (probability == 0.0) return sorted[0];
		int index = Math.max(0, (int) Math.ceil(probability * sorted.length) - 1);
		return sorted[Math.min(index, sorted.length - 1)];
	}

	@Override public double random() {
		return sorted[Math.min((int) (random.nextDouble() * sorted.length), sorted.length - 1)];
	}

	@Override public double atomProbability(double x) {
		return (upperBound(x) - lowerBound(x)) / (double) sorted.length;
	}

	@Override public double getLowerBound() { return sorted[0]; }
	@Override public double getUpperBound() { return sorted[sorted.length - 1]; }

	private int lowerBound(double value) {
		int low = 0, high = sorted.length;
		while (low < high) { int middle = (low + high) >>> 1;
			if (sorted[middle] < value) low = middle + 1; else high = middle; }
		return low;
	}
	private int upperBound(double value) {
		int low = 0, high = sorted.length;
		while (low < high) { int middle = (low + high) >>> 1;
			if (sorted[middle] <= value) low = middle + 1; else high = middle; }
		return low;
	}
}
