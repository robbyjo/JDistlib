/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Exact scalar law conditional on lower &lt; X &lt;= upper. */
public final class ConditionalDistribution extends GenericDistribution implements SupportedDistribution {
	private final GenericDistribution base;
	private final double lower, upper, lowerProbability, mass;
	public ConditionalDistribution(GenericDistribution base, double lower, double upper) {
		if (base == null || Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper))
			throw new IllegalArgumentException("base and ordered bounds required");
		this.base = base; this.lower = lower; this.upper = upper;
		this.lowerProbability = base.cumulative(lower, true, false);
		this.mass = base.cumulative(upper, true, false) - lowerProbability;
		if (!(mass > 0.0)) throw new IllegalArgumentException("conditioning event has zero probability");
	}
	@Override public double density(double x, boolean log) {
		if (x <= lower || x > upper) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double value = base.density(x, false) / mass; return log ? Math.log(value) : value;
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		double value = x <= lower ? 0.0 : x >= upper ? 1.0
				: (base.cumulative(x, true, false) - lowerProbability) / mass;
		if (!lowerTail) value = 1.0 - value; return logP ? Math.log(value) : value;
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (logP) p = Math.exp(p); if (!lowerTail) p = 1.0 - p;
		return base.quantile(lowerProbability + p * mass, true, false);
	}
	@Override public double random() { return quantile(random.nextDouble(), true, false); }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }
}
