/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Exact scalar law conditional on lower &lt; X &lt;= upper. */
public final class ConditionalDistribution extends GenericDistribution implements SupportedDistribution {
	private final GenericDistribution base;
	private final double lower, upper, logMass;
	private final boolean useSurvival;
	public ConditionalDistribution(GenericDistribution base, double lower, double upper) {
		if (base == null || Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper))
			throw new IllegalArgumentException("base and ordered bounds required");
		this.base = base; this.lower = lower; this.upper = upper;
		this.useSurvival = base.cumulative(lower, true, true) > -Math.log(2.0);
		this.logMass = logInterval(lower, upper);
		if (!Double.isFinite(logMass)) throw new IllegalArgumentException("conditioning event has zero or invalid probability");
	}
	@Override public double density(double x, boolean log) {
		if (x <= lower || x > upper) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double value = base.density(x, true) - logMass; return log ? value : Math.exp(value);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		double value = x <= lower ? (lowerTail ? Double.NEGATIVE_INFINITY : 0.0)
				: x >= upper ? (lowerTail ? 0.0 : Double.NEGATIVE_INFINITY)
				: (lowerTail ? logInterval(lower, x) : logInterval(x, upper)) - logMass;
		return logP ? value : Math.exp(value);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		double logLower = ProbabilityMath.logProbability(p, lowerTail, logP);
		if (Double.isNaN(logLower)) return Double.NaN;
		if (logLower == Double.NEGATIVE_INFINITY) return lower;
		if (logLower == 0.0) return upper;
		double target = useSurvival ? ProbabilityMath.add(base.cumulative(upper, false, true),
				ProbabilityMath.complement(logLower) + logMass)
				: ProbabilityMath.add(base.cumulative(lower, true, true), logLower + logMass);
		return base.quantile(target, !useSurvival, true);
	}
	@Override public double random() { return quantile(random.nextDouble(), true, false); }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }
	private double logInterval(double left, double right) {
		return useSurvival ? ProbabilityMath.subtract(base.cumulative(left, false, true),
				base.cumulative(right, false, true))
				: ProbabilityMath.subtract(base.cumulative(right, true, true), base.cumulative(left, true, true));
	}
}
