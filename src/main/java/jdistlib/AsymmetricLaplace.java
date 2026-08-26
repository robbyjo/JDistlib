/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Three-parameter asymmetric Laplace distribution used in quantile regression. */
public final class AsymmetricLaplace extends GenericDistribution
		implements SupportedDistribution {
	private final double location;
	private final double scale;
	private final double asymmetry;

	public AsymmetricLaplace(double location, double scale, double asymmetry) {
		this.location = location;
		this.scale = scale;
		this.asymmetry = asymmetry;
	}

	private static boolean invalid(double location, double scale,
			double asymmetry) {
		return !Double.isFinite(location) || !(scale > 0.0)
				|| !Double.isFinite(scale) || !(asymmetry > 0.0 && asymmetry < 1.0);
	}

	public static double density(double x, double location, double scale,
			double asymmetry, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(scale)
				|| Double.isNaN(asymmetry)) return x + location + scale + asymmetry;
		if (invalid(location, scale, asymmetry)) return Double.NaN;
		if (Double.isInfinite(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double z = (x - location) / scale;
		double value = Math.log(asymmetry) + Math.log1p(-asymmetry)
				- Math.log(scale) + (z < 0.0 ? (1.0 - asymmetry) * z
						: -asymmetry * z);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double location, double scale,
			double asymmetry, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(scale)
				|| Double.isNaN(asymmetry)) return x + location + scale + asymmetry;
		if (invalid(location, scale, asymmetry)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double z = (x - location) / scale;
		double value;
		if (z < 0.0) {
			double logLower = Math.log(asymmetry) + (1.0 - asymmetry) * z;
			value = lowerTail ? logLower : DistributionUtil.logOneMinusExp(logLower);
		} else {
			double logUpper = Math.log1p(-asymmetry) - asymmetry * z;
			value = lowerTail ? DistributionUtil.logOneMinusExp(logUpper) : logUpper;
		}
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double probability, double location,
			double scale, double asymmetry, boolean lowerTail, boolean logP) {
		if (Double.isNaN(probability) || Double.isNaN(location)
				|| Double.isNaN(scale) || Double.isNaN(asymmetry)) {
			return probability + location + scale + asymmetry;
		}
		if (invalid(location, scale, asymmetry)
				|| DistributionUtil.invalidProbability(probability, logP)) {
			return Double.NaN;
		}
		double logLower = lowerTail
				? (logP ? probability : Math.log(probability))
				: (logP ? DistributionUtil.logOneMinusExp(probability)
						: Math.log1p(-probability));
		double logUpper = lowerTail
				? (logP ? DistributionUtil.logOneMinusExp(probability)
						: Math.log1p(-probability))
				: (logP ? probability : Math.log(probability));
		if (logLower == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
		if (logUpper == Double.NEGATIVE_INFINITY) return Double.POSITIVE_INFINITY;
		if (logLower < Math.log(asymmetry)) {
			return location + scale * (logLower - Math.log(asymmetry))
					/ (1.0 - asymmetry);
		}
		return location - scale * (logUpper - Math.log1p(-asymmetry))
				/ asymmetry;
	}

	public static double random(double location, double scale, double asymmetry,
			RandomEngine random) {
		if (invalid(location, scale, asymmetry)) return Double.NaN;
		return quantile(random.nextDouble(), location, scale, asymmetry,
				true, false);
	}

	@Override public double density(double x, boolean log) {
		return density(x, location, scale, asymmetry, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, location, scale, asymmetry, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, location, scale, asymmetry, lowerTail, logP);
	}
	@Override public double random() { return random(location, scale, asymmetry, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
