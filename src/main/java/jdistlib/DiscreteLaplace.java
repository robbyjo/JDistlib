/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Discrete Laplace distribution on the lattice {@code location + Z}.
 *
 * <p>The probability of {@code location + k} is
 * {@code (1 - p) / (1 + p) * p^abs(k)} for integer {@code k} and
 * {@code 0 < p < 1}.</p>
 */
public final class DiscreteLaplace extends GenericDistribution
		implements SupportedDistribution {
	private final double location;
	private final double p;

	public DiscreteLaplace(double location, double p) {
		this.location = location;
		this.p = p;
	}

	private static boolean invalid(double location, double p) {
		return !Double.isFinite(location) || !(p > 0.0 && p < 1.0);
	}

	public static double density(double x, double location, double p,
			boolean log) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(p)) {
			return x + location + p;
		}
		if (invalid(location, p)) return Double.NaN;
		double k = x - location;
		if (!Double.isFinite(k) || k != Math.rint(k)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double value = Math.log1p(-p) - Math.log1p(p)
				+ Math.abs(k) * Math.log(p);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double location, double p,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(location) || Double.isNaN(p)) {
			return x + location + p;
		}
		if (invalid(location, p)) return Double.NaN;
		if (x == Double.NEGATIVE_INFINITY) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}

		double k = Math.floor(x - location);
		double logPValue;
		if (k < 0.0) {
			double logLower = -k * Math.log(p) - Math.log1p(p);
			logPValue = lowerTail ? logLower
					: DistributionUtil.logOneMinusExp(logLower);
		} else {
			double logUpper = (k + 1.0) * Math.log(p) - Math.log1p(p);
			logPValue = lowerTail ? DistributionUtil.logOneMinusExp(logUpper)
					: logUpper;
		}
		return logP ? logPValue : Math.exp(logPValue);
	}

	public static double quantile(double probability, double location, double p,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(probability) || Double.isNaN(location)
				|| Double.isNaN(p)) return probability + location + p;
		if (invalid(location, p)
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

		double logRatio = Math.log(p);
		double logNormalizer = Math.log1p(p);
		double lattice;
		if (logLower <= logRatio - logNormalizer) {
			lattice = -Math.floor((logLower + logNormalizer) / logRatio);
		} else if (logLower <= -logNormalizer) {
			lattice = 0.0;
		} else {
			lattice = Math.ceil((logUpper + logNormalizer) / logRatio) - 1.0;
		}

		// Correct the occasional one-unit rounding error at exact jump points.
		double value = location + lattice;
		if (cumulative(value, location, p, true, true) < logLower) {
			value += 1.0;
		} else if (cumulative(value - 1.0, location, p, true, true) >= logLower) {
			value -= 1.0;
		}
		return value;
	}

	public static double random(double location, double p, RandomEngine random) {
		if (invalid(location, p)) return Double.NaN;
		double success = 1.0 - p;
		return location + Geometric.random(success, random)
				- Geometric.random(success, random);
	}

	public static double[] random(int n, double location, double p,
			RandomEngine random) {
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = random(location, p, random);
		return values;
	}

	@Override public double density(double x, boolean log) {
		return density(x, location, p, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, location, p, lowerTail, logP);
	}
	@Override public double quantile(double probability, boolean lowerTail,
			boolean logP) {
		return quantile(probability, location, p, lowerTail, logP);
	}
	@Override public double random() { return random(location, p, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
