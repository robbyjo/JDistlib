/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.AtomAwareDistribution;
import jdistlib.generic.GenericDistribution;

/** Exact minimum or maximum of independent identically distributed variables. */
public final class OrderStatisticDistribution extends GenericDistribution
		implements SupportedDistribution, AtomAwareDistribution {
	private final GenericDistribution base;
	private final int count;
	private final boolean maximum;

	private OrderStatisticDistribution(GenericDistribution base, int count, boolean maximum) {
		if (base == null || count < 1) throw new IllegalArgumentException("base law and positive count required");
		this.base = base; this.count = count; this.maximum = maximum;
	}
	public static OrderStatisticDistribution maximum(GenericDistribution base, int count) {
		return new OrderStatisticDistribution(base, count, true);
	}
	public static OrderStatisticDistribution minimum(GenericDistribution base, int count) {
		return new OrderStatisticDistribution(base, count, false);
	}
	@Override public double density(double x, boolean log) {
		if (count == 1) return base.density(x, log);
		double logDensity;
		double mass = base instanceof AtomAwareDistribution ? ((AtomAwareDistribution) base).atomProbability(x) : 0.0;
		if (mass > 0.0) {
			double logF = base.cumulative(x, true, true), logS = base.cumulative(x, false, true);
			logDensity = maximum ? ProbabilityMath.subtract(count * logF,
					count * ProbabilityMath.subtract(logF, Math.log(mass)))
					: ProbabilityMath.subtract(count * ProbabilityMath.add(logS, Math.log(mass)), count * logS);
		} else {
			logDensity = Math.log(count) + base.density(x, true)
					+ (count - 1.0) * base.cumulative(x, maximum, true);
		}
		return log ? logDensity : Math.exp(logDensity);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		double opposite = base.cumulative(x, !maximum, true);
		if (opposite + Math.log(count) < -36.0 && lowerTail != maximum) {
			double value = opposite + Math.log(count);
			return logP ? value : Math.exp(value);
		}
		double value = count * base.cumulative(x, maximum, true);
		if (lowerTail != maximum) value = ProbabilityMath.complement(value);
		return logP ? value : Math.exp(value);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		double small = ProbabilityMath.logProbability(p, lowerTail != maximum, logP);
		if (small < -36.0) return base.quantile(small - Math.log(count), !maximum, true);
		double value = ProbabilityMath.logProbability(p, lowerTail == maximum, logP);
		return base.quantile(value / count, maximum, true);
	}
	@Override public double random() {
		return quantile(random.nextDouble(), true, false);
	}
	@Override public double getLowerBound() { return base instanceof SupportedDistribution ? ((SupportedDistribution) base).getLowerBound() : Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return base instanceof SupportedDistribution ? ((SupportedDistribution) base).getUpperBound() : Double.POSITIVE_INFINITY; }
	@Override public double atomProbability(double x) {
		return base instanceof AtomAwareDistribution && ((AtomAwareDistribution) base).atomProbability(x) > 0.0
				? density(x, false) : 0.0;
	}
}
