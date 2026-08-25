/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.Bessel;
import jdistlib.rng.RandomEngine;

/** Rice (Rician) distribution with scale {@code sigma} and distance {@code nu}. */
public class Rice extends GenericDistribution {
	private static boolean invalid(double sigma, double nu) {
		return !(sigma > 0.0) || nu < 0.0 || Double.isInfinite(sigma)
				|| Double.isInfinite(nu);
	}

	public static double density(double x, double sigma, double nu,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(sigma) || Double.isNaN(nu)) {
			return x + sigma + nu;
		}
		if (invalid(sigma, nu)) return Double.NaN;
		if (x <= 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double sigma2 = sigma * sigma;
		double argument = x * nu / sigma2;
		double scaledBessel = Bessel.i(argument, 0.0, true);
		double result = log(x) - log(sigma2)
				- (x - nu) * (x - nu) / (2.0 * sigma2) + log(scaledBessel);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double sigma, double nu,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(sigma) || Double.isNaN(nu)) {
			return x + sigma + nu;
		}
		if (invalid(sigma, nu)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		double ratio = x / sigma;
		double noncentrality = nu / sigma;
		return NonCentralChiSquare.cumulative(ratio * ratio, 2.0,
				noncentrality * noncentrality, lowerTail, logP);
	}

	public static double quantile(double p, double sigma, double nu,
			boolean lowerTail, boolean logP) {
		if (invalid(sigma, nu)) return Double.NaN;
		double noncentrality = nu / sigma;
		double value = NonCentralChiSquare.quantile(p, 2.0,
				noncentrality * noncentrality, lowerTail, logP);
		return sigma * sqrt(value);
	}

	public static double random(double sigma, double nu, RandomEngine random) {
		if (invalid(sigma, nu)) return Double.NaN;
		double first = Normal.random(nu, sigma, random);
		double second = Normal.random(0.0, sigma, random);
		return Math.hypot(first, second);
	}
	public static double[] random(int n, double sigma, double nu,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(sigma, nu, random);
		return result;
	}

	private final double sigma, nu;
	public Rice(double sigma, double nu) { this.sigma = sigma; this.nu = nu; }
	@Override public double density(double x, boolean logP) {
		return density(x, sigma, nu, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, sigma, nu, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, sigma, nu, lowerTail, logP);
	}
	@Override public double random() { return random(sigma, nu, random); }
}
