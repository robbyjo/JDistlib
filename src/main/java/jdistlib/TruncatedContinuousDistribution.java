/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;

/** Continuous distribution conditioned to lie in a nonempty interval. */
public final class TruncatedContinuousDistribution extends GenericDistribution
		implements SupportedDistribution {
	private final GenericDistribution base;
	private final double lower;
	private final double upper;
	private final double logRetainedProbability;
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
		logRetainedProbability = logInterval(lower, upper);
		retainedProbability = Math.exp(logRetainedProbability);
		if (!Double.isFinite(logRetainedProbability)) {
			throw new IllegalArgumentException("truncation interval has no finite positive mass");
		}
	}

	private double logInterval(double a, double b) {
		double upperCdf = base.cumulative(b, true, true);
		if (upperCdf <= -Math.log(2.0)) {
			return upperCdf + DistributionUtil.logOneMinusExp(
					base.cumulative(a, true, true) - upperCdf);
		}
		double lowerSurvival = base.cumulative(a, false, true);
		return lowerSurvival + DistributionUtil.logOneMinusExp(
				base.cumulative(b, false, true) - lowerSurvival);
	}

	public GenericDistribution getBaseDistribution() { return base; }
	public double getRetainedProbability() { return retainedProbability; }
	/** Log retained mass, also available when the ordinary mass underflows. */
	public double getLogRetainedProbability() { return logRetainedProbability; }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < lower || x > upper) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double value = base.density(x, true) - logRetainedProbability;
		return log ? value : Math.exp(value);
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x <= lower) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x >= upper) return DistributionUtil.boundary(true, lowerTail, logP);
		double value = Math.min(0.0, (lowerTail ? logInterval(lower, x)
				: logInterval(x, upper)) - logRetainedProbability);
		return logP ? value : Math.exp(value);
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double requested = logP ? p : Math.log(p);
		double complement = logP ? DistributionUtil.logOneMinusExp(p) : Math.log1p(-p);
		double logLower = lowerTail ? requested : complement;
		double logUpper = lowerTail ? complement : requested;
		if (logLower == Double.NEGATIVE_INFINITY) return lower;
		if (logUpper == Double.NEGATIVE_INFINITY) return upper;
		double targetLower = DistributionUtil.logAdd(base.cumulative(lower, true, true), logLower + logRetainedProbability);
		double targetUpper = DistributionUtil.logAdd(base.cumulative(upper, false, true), logUpper + logRetainedProbability);
		double result = targetLower < targetUpper ? base.quantile(targetLower, true, true)
				: base.quantile(targetUpper, false, true);
		return Math.max(lower, Math.min(upper, result));
	}

	@Override public double random() { return quantile(random.nextDouble()); }
}
