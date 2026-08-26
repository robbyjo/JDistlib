/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Distribution of an independent normal variate plus an exponential variate. */
public final class ExponentiallyModifiedGaussian extends GenericDistribution
		implements SupportedDistribution {
	private final double normalMean;
	private final double normalStandardDeviation;
	private final double exponentialRate;

	public ExponentiallyModifiedGaussian(double normalMean,
			double normalStandardDeviation, double exponentialRate) {
		this.normalMean = normalMean;
		this.normalStandardDeviation = normalStandardDeviation;
		this.exponentialRate = exponentialRate;
	}

	private static boolean invalid(double mean, double standardDeviation,
			double rate) {
		return !Double.isFinite(mean) || !(standardDeviation > 0.0)
				|| !Double.isFinite(standardDeviation) || !(rate > 0.0)
				|| !Double.isFinite(rate);
	}

	public static double density(double x, double mean, double standardDeviation,
			double rate, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(mean)
				|| Double.isNaN(standardDeviation) || Double.isNaN(rate)) {
			return x + mean + standardDeviation + rate;
		}
		if (invalid(mean, standardDeviation, rate)) return Double.NaN;
		if (Double.isInfinite(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double z = (x - mean) / standardDeviation;
		double scaledRate = rate * standardDeviation;
		double value = Math.log(rate) - rate * (x - mean)
				+ 0.5 * scaledRate * scaledRate
				+ Normal.cumulative(z - scaledRate, 0.0, 1.0, true, true);
		return log ? value : Math.exp(value);
	}

	private static double logLower(double x, double mean, double standardDeviation,
			double rate) {
		double z = (x - mean) / standardDeviation;
		double scaledRate = rate * standardDeviation;
		double logNormal = Normal.cumulative(z, 0.0, 1.0, true, true);
		double logCorrection = -rate * (x - mean)
				+ 0.5 * scaledRate * scaledRate
				+ Normal.cumulative(z - scaledRate, 0.0, 1.0, true, true);
		double ratio = logCorrection - logNormal;
		if (!(ratio < 0.0)) {
			// The Mills-ratio limit avoids a cancellation artifact in the far left tail.
			return logNormal + Math.log(scaledRate
					/ (Math.abs(z) + scaledRate));
		}
		return logNormal + DistributionUtil.logOneMinusExp(ratio);
	}

	private static double logUpper(double x, double mean, double standardDeviation,
			double rate) {
		double z = (x - mean) / standardDeviation;
		double scaledRate = rate * standardDeviation;
		double normalTail = Normal.cumulative(z, 0.0, 1.0, false, true);
		double convolutionTail = -rate * (x - mean)
				+ 0.5 * scaledRate * scaledRate
				+ Normal.cumulative(z - scaledRate, 0.0, 1.0, true, true);
		return DistributionUtil.logAdd(normalTail, convolutionTail);
	}

	public static double cumulative(double x, double mean, double standardDeviation,
			double rate, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mean)
				|| Double.isNaN(standardDeviation) || Double.isNaN(rate)) {
			return x + mean + standardDeviation + rate;
		}
		if (invalid(mean, standardDeviation, rate)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double value = lowerTail ? logLower(x, mean, standardDeviation, rate)
				: logUpper(x, mean, standardDeviation, rate);
		return logP ? value : Math.exp(value);
	}

	public static double quantile(double probability, double mean,
			double standardDeviation, double rate, boolean lowerTail, boolean logP) {
		if (Double.isNaN(probability) || invalid(mean, standardDeviation, rate)
				|| DistributionUtil.invalidProbability(probability, logP)) {
			return Double.NaN;
		}
		double logTarget = lowerTail
				? (logP ? probability : Math.log(probability))
				: (logP ? DistributionUtil.logOneMinusExp(probability)
						: Math.log1p(-probability));
		double logOpposite = lowerTail
				? (logP ? DistributionUtil.logOneMinusExp(probability)
						: Math.log1p(-probability))
				: (logP ? probability : Math.log(probability));
		if (logTarget == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
		if (logOpposite == Double.NEGATIVE_INFINITY) return Double.POSITIVE_INFINITY;

		double center = mean + 1.0 / rate;
		double step = standardDeviation + 1.0 / rate;
		double low = center - step;
		double high = center + step;
		while (logLower(low, mean, standardDeviation, rate) >= logTarget) {
			high = low;
			step *= 2.0;
			low = center - step;
		}
		while (logLower(high, mean, standardDeviation, rate) < logTarget) {
			low = high;
			step *= 2.0;
			high = center + step;
		}
		for (int i = 0; i < 140; i++) {
			double middle = low + (high - low) * 0.5;
			if (logLower(middle, mean, standardDeviation, rate) >= logTarget) {
				high = middle;
			} else low = middle;
		}
		return low + (high - low) * 0.5;
	}

	public static double random(double mean, double standardDeviation, double rate,
			RandomEngine random) {
		if (invalid(mean, standardDeviation, rate)) return Double.NaN;
		return Normal.random(mean, standardDeviation, random)
				+ Exponential.random(1.0 / rate, random);
	}

	@Override public double density(double x, boolean log) {
		return density(x, normalMean, normalStandardDeviation, exponentialRate, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, normalMean, normalStandardDeviation, exponentialRate,
				lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, normalMean, normalStandardDeviation, exponentialRate,
				lowerTail, logP);
	}
	@Override public double random() {
		return random(normalMean, normalStandardDeviation, exponentialRate, random);
	}
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
