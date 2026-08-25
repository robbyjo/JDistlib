/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Maxwell distribution using VGAM's positive rate parameterization.
 * See {@link MaxwellBoltzmann} for the conventional scale parameterization.
 */
public class Maxwell extends GenericDistribution {
	private static boolean invalid(double rate) {
		return !(rate > 0.0) || Double.isInfinite(rate);
	}

	public static double density(double x, double rate, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(rate)) return x + rate;
		if (invalid(rate)) return Double.NaN;
		if (x <= 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = 0.5 * log(2.0 / Math.PI) + 1.5 * log(rate)
				+ 2.0 * log(x) - 0.5 * rate * x * x;
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double rate, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(rate)) return x + rate;
		if (invalid(rate)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		return Gamma.cumulative(0.5 * rate * x * x, 1.5, 1.0,
				lowerTail, logP);
	}

	public static double quantile(double p, double rate, boolean lowerTail,
			boolean logP) {
		if (invalid(rate)) return Double.NaN;
		return sqrt(2.0 * Gamma.quantile(p, 1.5, 1.0, lowerTail, logP) / rate);
	}

	public static double random(double rate, RandomEngine random) {
		if (invalid(rate)) return Double.NaN;
		return sqrt(2.0 * Gamma.random(1.5, 1.0, random) / rate);
	}
	public static double[] random(int n, double rate, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(rate, random);
		return result;
	}

	private final double rate;
	public Maxwell(double rate) { this.rate = rate; }
	@Override public double density(double x, boolean logP) {
		return density(x, rate, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, rate, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, rate, lowerTail, logP);
	}
	@Override public double random() { return random(rate, random); }
}
