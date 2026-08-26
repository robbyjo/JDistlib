/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Huber least-favourable distribution with Gaussian center and exponential tails. */
public final class Huber extends GenericDistribution implements SupportedDistribution {
	private static final double LOG_TWO = Math.log(2.0);
	private final double location;
	private final double scale;
	private final double threshold;

	public Huber(double location, double scale, double threshold) {
		this.location = location;
		this.scale = scale;
		this.threshold = threshold;
	}

	private static boolean invalid(double location, double scale, double threshold) {
		return !Double.isFinite(location) || !(scale > 0.0)
				|| !Double.isFinite(scale) || !(threshold > 0.0)
				|| !Double.isFinite(threshold);
	}

	private static double logArea(double threshold) {
		double normalPart = Normal.cumulative(threshold, 0.0, 1.0, true, false) - 0.5;
		double logNormalPart = normalPart > 0.0 ? Math.log(normalPart)
				: Double.NEGATIVE_INFINITY;
		double logTailPart = Normal.density(threshold, 0.0, 1.0, true)
				- Math.log(threshold);
		return LOG_TWO + DistributionUtil.logAdd(logNormalPart, logTailPart);
	}

	private static double logSmallerTail(double absoluteZ, double threshold) {
		double area = logArea(threshold);
		if (absoluteZ >= threshold) {
			return 0.5 * threshold * threshold - threshold * absoluteZ
					+ Normal.density(0.0, 0.0, 1.0, true)
					- Math.log(threshold) - area;
		}
		double logPhiAtZ = Normal.cumulative(-absoluteZ, 0.0, 1.0, true, true);
		double logPhiAtThreshold = Normal.cumulative(-threshold, 0.0, 1.0,
				true, true);
		double logDifference = logPhiAtZ + DistributionUtil.logOneMinusExp(
				logPhiAtThreshold - logPhiAtZ);
		double logBoundaryTail = Normal.density(threshold, 0.0, 1.0, true)
				- Math.log(threshold);
		return DistributionUtil.logAdd(logBoundaryTail, logDifference) - area;
	}

	public static double density(double x, double location, double scale,
			double threshold, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(scale)
				|| Double.isNaN(threshold)) return x + location + scale + threshold;
		if (invalid(location, scale, threshold)) return Double.NaN;
		if (Double.isInfinite(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double z = Math.abs((x - location) / scale);
		double loss = z <= threshold ? 0.5 * z * z
				: threshold * z - 0.5 * threshold * threshold;
		double value = -loss + Normal.density(0.0, 0.0, 1.0, true)
				- logArea(threshold) - Math.log(scale);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double location, double scale,
			double threshold, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(scale)
				|| Double.isNaN(threshold)) return x + location + scale + threshold;
		if (invalid(location, scale, threshold)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double z = (x - location) / scale;
		if (z == 0.0) return logP ? -LOG_TWO : 0.5;
		double logSmall = logSmallerTail(Math.abs(z), threshold);
		double logLower = z <= 0.0 ? logSmall
				: DistributionUtil.logOneMinusExp(logSmall);
		double logUpper = z <= 0.0 ? DistributionUtil.logOneMinusExp(logSmall)
				: logSmall;
		double value = lowerTail ? logLower : logUpper;
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double probability, double location,
			double scale, double threshold, boolean lowerTail, boolean logP) {
		if (Double.isNaN(probability) || Double.isNaN(location)
				|| Double.isNaN(scale) || Double.isNaN(threshold)) {
			return probability + location + scale + threshold;
		}
		if (invalid(location, scale, threshold)
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

		boolean left = logLower <= -LOG_TWO;
		double logSmall = left ? logLower : logUpper;
		double area = logArea(threshold);
		double logBoundary = Normal.density(threshold, 0.0, 1.0, true)
				- Math.log(threshold) - area;
		double negativeZ;
		if (logSmall <= logBoundary) {
			negativeZ = (Math.log(threshold) + logSmall + area
					- Normal.density(0.0, 0.0, 1.0, true)) / threshold
					- 0.5 * threshold;
		} else {
			double logTailComponent = Normal.density(threshold, 0.0, 1.0, true)
					- Math.log(threshold);
			double logPhiThreshold = Normal.cumulative(-threshold, 0.0, 1.0,
					true, true);
			double logDifference = logTailComponent
					+ DistributionUtil.logOneMinusExp(logPhiThreshold - logTailComponent);
			double logProduct = logSmall + area;
			double logTarget = logProduct + DistributionUtil.logOneMinusExp(
					logDifference - logProduct);
			negativeZ = Normal.quantile(logTarget, 0.0, 1.0, true, true);
		}
		return location + (left ? negativeZ : -negativeZ) * scale;
	}

	public static double random(double location, double scale, double threshold,
			RandomEngine random) {
		if (invalid(location, scale, threshold)) return Double.NaN;
		return quantile(random.nextDouble(), location, scale, threshold,
				true, false);
	}

	@Override public double density(double x, boolean log) {
		return density(x, location, scale, threshold, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, location, scale, threshold, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, location, scale, threshold, lowerTail, logP);
	}
	@Override public double random() { return random(location, scale, threshold, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
