/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;

/** Normalized finite mixture of scalar distribution objects. */
public final class MixtureDistribution extends GenericDistribution
		implements SupportedDistribution {
	private static final int QUANTILE_ITERATIONS = 160;
	private final GenericDistribution[] components;
	private final double[] weights;
	private final double[] cumulativeWeights;
	private final double lower;
	private final double upper;

	public MixtureDistribution(double[] weights,
			GenericDistribution... components) {
		if (weights == null || components == null || components.length == 0
				|| weights.length != components.length) {
			throw new IllegalArgumentException(
					"matching nonempty weights and components are required");
		}
		this.components = components.clone();
		this.weights = weights.clone();
		double sum = 0.0;
		for (int i = 0; i < weights.length; i++) {
			if (components[i] == null || !(weights[i] >= 0.0)
					|| !Double.isFinite(weights[i])) {
				throw new IllegalArgumentException("components and weights must be valid");
			}
			sum += weights[i];
		}
		if (!(sum > 0.0) || !Double.isFinite(sum)) {
			throw new IllegalArgumentException("mixture weights must have a finite positive sum");
		}
		cumulativeWeights = new double[weights.length];
		double cumulative = 0.0;
		double low = Double.POSITIVE_INFINITY;
		double high = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < weights.length; i++) {
			this.weights[i] /= sum;
			cumulative += this.weights[i];
			cumulativeWeights[i] = cumulative;
			if (this.weights[i] == 0.0) continue;
			double componentLower = components[i] instanceof SupportedDistribution
					? ((SupportedDistribution) components[i]).getLowerBound()
					: components[i].quantile(0.0);
				double componentUpper = components[i] instanceof SupportedDistribution
					? ((SupportedDistribution) components[i]).getUpperBound()
					: components[i].quantile(1.0);
			if (Double.isNaN(componentLower) || Double.isNaN(componentUpper)
					|| componentLower > componentUpper) {
				throw new IllegalArgumentException(
						"positive-weight components must have valid ordered support bounds");
			}
			low = Math.min(low, componentLower);
			high = Math.max(high, componentUpper);
		}
		cumulativeWeights[cumulativeWeights.length - 1] = 1.0;
		lower = low;
		upper = high;
	}

	public GenericDistribution[] getComponents() { return components.clone(); }
	public double[] getWeights() { return weights.clone(); }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (!log) {
			double sum = 0.0;
			for (int i = 0; i < components.length; i++) {
				if (weights[i] == 0.0) continue;
				sum += weights[i] * components[i].density(x, false);
			}
			return sum;
		}
		double result = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < components.length; i++) {
			if (weights[i] == 0.0) continue;
			result = logAdd(result, Math.log(weights[i])
					+ components[i].density(x, true));
		}
		return result;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		double probability = 0.0;
		for (int i = 0; i < components.length; i++) {
			probability += weights[i]
					* components[i].cumulative(x, lowerTail, false);
		}
		probability = Math.max(0.0, Math.min(1.0, probability));
		return logP ? Math.log(probability) : probability;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		double target = lowerTail ? probability : 1.0 - probability;
		if (target <= 0.0) return lower;
		if (target >= 1.0) return upper;
		if (Double.isFinite(lower)
				&& cumulative(lower, true, false) >= target) return lower;
		double low = Double.isFinite(lower) ? lower : -1.0;
		double high = Double.isFinite(upper) ? upper : 1.0;
		for (int i = 0; !Double.isFinite(lower)
				&& cumulative(low, true, false) >= target && i < 1024; i++) {
			high = low;
			low = low < 0.0 ? low * 2.0 : -1.0;
		}
		for (int i = 0; !Double.isFinite(upper)
				&& cumulative(high, true, false) < target && i < 1024; i++) {
			low = high;
			high = high > 0.0 ? high * 2.0 : 1.0;
		}
		if (cumulative(high, true, false) < target) return Double.NaN;
		for (int i = 0; i < QUANTILE_ITERATIONS; i++) {
			double middle = low * 0.5 + high * 0.5;
			if (middle == low || middle == high) break;
			if (cumulative(middle, true, false) >= target) high = middle;
			else low = middle;
		}
		return high;
	}

	@Override public double random() {
		double selector = random.nextDouble();
		int index = 0;
		while (index + 1 < cumulativeWeights.length
				&& selector >= cumulativeWeights[index]) index++;
		return components[index].quantile(random.nextDouble());
	}

	private static double logAdd(double x, double y) {
		if (x == Double.NEGATIVE_INFINITY) return y;
		if (y == Double.NEGATIVE_INFINITY) return x;
		double high = Math.max(x, y);
		return high + Math.log1p(Math.exp(Math.min(x, y) - high));
	}
}
