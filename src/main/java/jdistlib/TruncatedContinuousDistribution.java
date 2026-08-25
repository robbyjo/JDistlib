/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;

/** Continuous distribution conditioned to lie in a nonempty interval. */
public final class TruncatedContinuousDistribution extends GenericDistribution
		implements SupportedDistribution {
	private final GenericDistribution base;
	private final double lower;
	private final double upper;
	private final double baseLowerProbability;
	private final double retainedProbability;

	public TruncatedContinuousDistribution(GenericDistribution base, double lower,
			double upper) {
		if (base == null || Double.isNaN(lower) || Double.isNaN(upper)
				|| !(lower < upper)) {
			throw new IllegalArgumentException("base and ordered truncation bounds are required");
		}
		this.base = base;
		this.lower = lower;
		this.upper = upper;
		baseLowerProbability = lower == Double.NEGATIVE_INFINITY ? 0.0
				: base.cumulative(lower, true, false);
		double baseUpperProbability = upper == Double.POSITIVE_INFINITY ? 1.0
				: base.cumulative(upper, true, false);
		retainedProbability = baseUpperProbability - baseLowerProbability;
		if (!(retainedProbability > 0.0) || !Double.isFinite(retainedProbability)) {
			throw new IllegalArgumentException("truncation interval has no finite positive mass");
		}
	}

	public GenericDistribution getBaseDistribution() { return base; }
	public double getRetainedProbability() { return retainedProbability; }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < lower || x > upper) return log ? Double.NEGATIVE_INFINITY : 0.0;
		return log ? base.density(x, true) - Math.log(retainedProbability)
				: base.density(x, false) / retainedProbability;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		double probability;
		if (x <= lower) probability = 0.0;
		else if (x >= upper) probability = 1.0;
		else probability = (base.cumulative(x, true, false)
				- baseLowerProbability) / retainedProbability;
		probability = Math.max(0.0, Math.min(1.0, probability));
		double requested = lowerTail ? probability : 1.0 - probability;
		return logP ? Math.log(requested) : requested;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		double target = lowerTail ? probability : 1.0 - probability;
		if (target <= 0.0) return lower;
		if (target >= 1.0) return upper;
		return Math.max(lower, Math.min(upper, base.quantile(
				baseLowerProbability + target * retainedProbability, true, false)));
	}

	@Override public double random() { return quantile(random.nextDouble()); }
}
