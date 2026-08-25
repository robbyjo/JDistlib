/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;

/** Winsorized/censored scalar distribution with explicit atoms at both bounds. */
public final class CensoredDistribution extends GenericDistribution
		implements SupportedDistribution, AtomAwareDistribution {
	private final GenericDistribution base;
	private final double lower;
	private final double upper;
	private final double lowerMass;
	private final double upperMass;

	public CensoredDistribution(GenericDistribution base, double lower, double upper) {
		if (base == null || !Double.isFinite(lower) || !Double.isFinite(upper)
				|| !(lower < upper)) {
			throw new IllegalArgumentException("finite ordered censoring bounds are required");
		}
		this.base = base;
		this.lower = lower;
		this.upper = upper;
		lowerMass = base.cumulative(lower, true, false);
		upperMass = base.cumulative(Math.nextDown(upper), false, false);
	}

	public GenericDistribution getBaseDistribution() { return base; }
	public double getLowerAtomProbability() { return lowerMass; }
	public double getUpperAtomProbability() { return upperMass; }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }

	@Override public double density(double x, boolean log) {
		double value;
		if (Double.isNaN(x)) return Double.NaN;
		if (x == lower) value = lowerMass;
		else if (x == upper) value = upperMass;
		else if (x > lower && x < upper) value = base.density(x, false);
		else value = 0.0;
		return log ? Math.log(value) : value;
	}

	@Override public double atomProbability(double x) {
		if (x == lower) return lowerMass;
		if (x == upper) return upperMass;
		return x > lower && x < upper && base instanceof AtomAwareDistribution
				? ((AtomAwareDistribution) base).atomProbability(x) : 0.0;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		double probability = x < lower ? 0.0
				: (x >= upper ? 1.0 : base.cumulative(x, true, false));
		double requested = lowerTail ? probability : 1.0 - probability;
		return logP ? Math.log(requested) : requested;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		double value = base.quantile(p, lowerTail, logP);
		return Double.isNaN(value) ? value : Math.max(lower, Math.min(upper, value));
	}

	@Override public double random() {
		return Math.max(lower, Math.min(upper, base.quantile(random.nextDouble())));
	}
}
