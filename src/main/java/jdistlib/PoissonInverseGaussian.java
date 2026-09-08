/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Poisson-inverse Gaussian distribution from {@code actuar}.
 *
 * <p>{@code mean} is the mean of the inverse-Gaussian mixing variable and
 * {@code dispersion} is its dispersion. The count variance is
 * {@code mean + dispersion * mean^3}.</p>
 */
public class PoissonInverseGaussian extends GenericDistribution {
	private static boolean invalid(double mean, double dispersion) {
		return !(mean > 0.0) || !(dispersion > 0.0);
	}

	private static double logMass(int x, double mean, double dispersion) {
		double t = 2.0 * dispersion * mean * mean;
		double p0 = Double.isInfinite(mean) ? -sqrt(2.0 / dispersion)
				: -2.0 * mean / (1.0 + sqrt(1.0 + t));
		if (x == 0) return p0;
		double previous = Double.isInfinite(mean) ? p0 - 0.5 * log(2.0 * dispersion)
				: p0 + log(mean) - 0.5 * log1p(t);
		double first = Double.isInfinite(mean) ? 1.0 : t / (1.0 + t);
		double second = Double.isInfinite(mean) ? 0.5 / dispersion : mean * mean / (1.0 + t);
		for (int i = 2; i <= x; i++) {
			double current = DistributionUtil.logAdd(log(first) + log1p(-1.5 / i) + previous,
					log(second) - log(i) - log(i - 1.0) + p0);
			p0 = previous; previous = current;
		}
		return previous;
	}

	private static double logUpperCumulative(int x, double mean, double dispersion) {
		double t = 2.0 * dispersion * mean * mean;
		double first = t / (1.0 + t), second = mean * mean / (1.0 + t);
		double previous2 = logMass(x, mean, dispersion);
		double previous1 = logMass(x + 1, mean, dispersion);
		double sum = previous1;
		for (long i = (long) x + 2; i < (long) x + 1000000; i++) {
			double term = DistributionUtil.logAdd(log(first) + log1p(-1.5 / i) + previous1,
					log(second) - log(i) - log(i - 1.0) + previous2);
			sum = DistributionUtil.logAdd(sum, term);
			if (term < sum - 37.0 && term < previous1) return sum;
			previous2 = previous1; previous1 = term;
		}
		return Double.NaN;
	}

	private static double mass(int x, double mean, double dispersion) {
		if (Double.isInfinite(mean)) {
			double logP0 = -sqrt(2.0 / dispersion);
			if (x == 0) return exp(logP0);
			double previous2 = exp(logP0);
			double previous1 = exp(logP0 - 0.5 * log(2.0 * dispersion));
			if (x == 1) return previous1;
			for (int i = 2; i <= x; i++) {
				double current = (1.0 - 1.5 / i) * previous1
						+ previous2 / (2.0 * dispersion * i * (i - 1.0));
				previous2 = previous1;
				previous1 = current;
			}
			return previous1;
		}
		double twicePhiMu2 = 2.0 * dispersion * mean * mean;
		double logP0 = -2.0 * mean / (1.0 + sqrt(1.0 + twicePhiMu2));
		if (x == 0) return exp(logP0);
		double previous2 = exp(logP0);
		double previous1 = exp(log(mean) + logP0 - 0.5 * log1p(twicePhiMu2));
		if (x == 1) return previous1;
		double first = 1.0 / (1.0 + 1.0 / twicePhiMu2);
		double second = mean * mean / (1.0 + twicePhiMu2);
		for (int i = 2; i <= x; i++) {
			double current = first * (1.0 - 1.5 / i) * previous1
					+ second * previous2 / (i * (i - 1.0));
			previous2 = previous1;
			previous1 = current;
		}
		return previous1;
	}

	private static double lowerCumulative(int x, double mean, double dispersion) {
		if (Double.isInfinite(mean)) {
			double logP0 = -sqrt(2.0 / dispersion);
			double previous2 = exp(logP0);
			double sum = previous2;
			if (x == 0) return sum;
			double previous1 = exp(logP0 - 0.5 * log(2.0 * dispersion));
			sum += previous1;
			for (int i = 2; i <= x; i++) {
				double current = (1.0 - 1.5 / i) * previous1
						+ previous2 / (2.0 * dispersion * i * (i - 1.0));
				sum += current;
				previous2 = previous1;
				previous1 = current;
			}
			return Math.min(1.0, sum);
		}
		double twicePhiMu2 = 2.0 * dispersion * mean * mean;
		double logP0 = -2.0 * mean / (1.0 + sqrt(1.0 + twicePhiMu2));
		double previous2 = exp(logP0);
		double sum = previous2;
		if (x == 0) return sum;
		double previous1 = exp(log(mean) + logP0 - 0.5 * log1p(twicePhiMu2));
		sum += previous1;
		double first = 1.0 / (1.0 + 1.0 / twicePhiMu2);
		double second = mean * mean / (1.0 + twicePhiMu2);
		for (int i = 2; i <= x; i++) {
			double current = first * (1.0 - 1.5 / i) * previous1
					+ second * previous2 / (i * (i - 1.0));
			sum += current;
			previous2 = previous1;
			previous1 = current;
		}
		return Math.min(1.0, sum);
	}

	public static double density(double x, double mean, double dispersion,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(dispersion)) {
			return x + mean + dispersion;
		}
		if (invalid(mean, dispersion)) return Double.NaN;
		if (Double.isInfinite(dispersion)) {
			double result = x == 0.0 ? 1.0 : 0.0;
			return giveLog ? log(result) : result;
		}
		if (x < 0.0 || x != Math.rint(x) || Double.isInfinite(x)
				|| x > Integer.MAX_VALUE) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		return giveLog ? logMass((int) x, mean, dispersion) : mass((int) x, mean, dispersion);
	}

	public static double cumulative(double x, double mean, double dispersion,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mean) || Double.isNaN(dispersion)) {
			return x + mean + dispersion;
		}
		if (invalid(mean, dispersion)) return Double.NaN;
		if (x < 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (Double.isInfinite(dispersion)) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		if (x > Integer.MAX_VALUE) return DistributionUtil.boundary(true,
				lowerTail, logP);
		int last = (int) Math.floor(x);
		double result = lowerCumulative(last, mean, dispersion);
		double t = 2.0 * dispersion * mean * mean;
		if (result > 0.5 && Double.isFinite(mean) && t < 1000.0 && last < Integer.MAX_VALUE - 1) {
			double upper = logUpperCumulative(last, mean, dispersion);
			double value = lowerTail ? DistributionUtil.logOneMinusExp(upper) : upper;
			return logP ? value : exp(value);
		}
		if (!lowerTail) result = Math.max(0.0, 1.0 - result);
		return logP ? log(result) : result;
	}

	public static double quantile(double p, double mean, double dispersion,
			boolean lowerTail, boolean logP) {
		if (invalid(mean, dispersion) || Double.isNaN(p)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		if (Double.isInfinite(dispersion)) return 0.0;
		return DistributionUtil.discreteQuantile(p, lowerTail, logP, 0.0,
				Double.POSITIVE_INFINITY,
				(x, lt, lp) -> cumulative(x, mean, dispersion, lt, lp));
	}

	public static double random(double mean, double dispersion,
			RandomEngine random) {
		if (invalid(mean, dispersion)) return Double.NaN;
		if (Double.isInfinite(dispersion)) return 0.0;
		double lambda = Double.isInfinite(mean)
				? 1.0 / dispersion / ChiSquare.random(1.0, random)
				: InvNormal.random(mean, sqrt(dispersion), random);
		return Poisson.random(lambda, random);
	}

	public static double[] random(int n, double mean, double dispersion,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mean, dispersion, random);
		return result;
	}

	private final double mean, dispersion;
	public PoissonInverseGaussian(double mean, double dispersion) {
		this.mean = mean; this.dispersion = dispersion;
	}
	@Override public double density(double x, boolean logP) {
		return density(x, mean, dispersion, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mean, dispersion, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mean, dispersion, lowerTail, logP);
	}
	@Override public double random() { return random(mean, dispersion, random); }
}
