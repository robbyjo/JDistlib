/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Exact minimum or maximum of independent identically distributed variables. */
public final class OrderStatisticDistribution extends GenericDistribution
		implements SupportedDistribution {
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
		double f = base.density(x, false);
		double p = base.cumulative(x, maximum, false);
		double value = count * f * Math.pow(p, count - 1.0);
		return log ? Math.log(value) : value;
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		double p = base.cumulative(x, true, false);
		double value = maximum ? Math.pow(p, count) : 1.0 - Math.pow(1.0 - p, count);
		if (!lowerTail) value = 1.0 - value;
		return logP ? Math.log(value) : value;
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (logP) p = Math.exp(p); if (!lowerTail) p = 1.0 - p;
		double baseP = maximum ? Math.pow(p, 1.0 / count) : 1.0 - Math.pow(1.0 - p, 1.0 / count);
		return base.quantile(baseP, true, false);
	}
	@Override public double random() {
		double answer = inverseSample();
		for (int i = 1; i < count; i++) answer = maximum ? Math.max(answer, inverseSample()) : Math.min(answer, inverseSample());
		return answer;
	}
	@Override public double getLowerBound() { return base instanceof SupportedDistribution ? ((SupportedDistribution) base).getLowerBound() : Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return base instanceof SupportedDistribution ? ((SupportedDistribution) base).getUpperBound() : Double.POSITIVE_INFINITY; }
	private double inverseSample() { return base.quantile(random.nextDouble(), true, false); }
}
