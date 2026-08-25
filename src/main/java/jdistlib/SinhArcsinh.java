/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.sqrt;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Four-parameter sinh-arcsinh distribution of Jones and Pewsey.
 *
 * <p>The {@code nu} and {@code tau} parameters independently control the two
 * tails; {@code nu = tau = 1} gives a normal distribution.</p>
 */
public class SinhArcsinh extends GenericDistribution {
	private static boolean invalid(double mu, double sigma, double nu, double tau) {
		return Double.isNaN(mu) || Double.isNaN(sigma) || Double.isNaN(nu)
				|| Double.isNaN(tau) || Double.isInfinite(mu)
				|| !(sigma > 0.0) || Double.isInfinite(sigma)
				|| !(nu > 0.0) || Double.isInfinite(nu)
				|| !(tau > 0.0) || Double.isInfinite(tau);
	}

	private static double asinh(double x) {
		double absolute = Math.abs(x);
		double result = absolute > 1.0e154 ? log(absolute) + log(2.0)
				: log(absolute + sqrt(absolute * absolute + 1.0));
		return Math.copySign(result, x);
	}

	private static double transformed(double w, double nu, double tau) {
		return 0.5 * (exp(tau * w) - exp(-nu * w));
	}

	public static double density(double x, double mu, double sigma, double nu,
			double tau, boolean giveLog) {
		if (Double.isNaN(x)) return x;
		if (invalid(mu, sigma, nu, tau)) return Double.NaN;
		if (Double.isInfinite(x)) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		double z = (x - mu) / sigma;
		double w = asinh(z);
		double expTau = exp(tau * w);
		double expNu = exp(-nu * w);
		double r = 0.5 * (expTau - expNu);
		double derivative = 0.5 * (tau * expTau + nu * expNu);
		double result;
		if (Double.isInfinite(r) || Double.isInfinite(derivative)) {
			result = Double.NEGATIVE_INFINITY;
		} else {
			result = -log(sigma) - 0.5 * log(2.0 * Math.PI)
					- 0.5 * Math.log1p(z * z) + log(derivative) - 0.5 * r * r;
		}
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double mu, double sigma, double nu,
			double tau, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return x;
		if (invalid(mu, sigma, nu, tau)) return Double.NaN;
		double r = transformed(asinh((x - mu) / sigma), nu, tau);
		return Normal.cumulative(r, 0.0, 1.0, lowerTail, logP);
	}

	public static double quantile(double p, double mu, double sigma, double nu,
			double tau, boolean lowerTail, boolean logP) {
		if (invalid(mu, sigma, nu, tau)
				|| DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
		double target = Normal.quantile(p, 0.0, 1.0, lowerTail, logP);
		if (Double.isInfinite(target)) return target;
		double low = -1.0;
		double high = 1.0;
		while (transformed(low, nu, tau) > target) low *= 2.0;
		while (transformed(high, nu, tau) < target) high *= 2.0;
		for (int i = 0; i < 120; i++) {
			double middle = (low + high) / 2.0;
			if (transformed(middle, nu, tau) < target) low = middle;
			else high = middle;
		}
		return mu + sigma * Math.sinh((low + high) / 2.0);
	}

	public static double random(double mu, double sigma, double nu, double tau,
			RandomEngine random) {
		return quantile(random.nextDouble(), mu, sigma, nu, tau, true, false);
	}

	public static double[] random(int n, double mu, double sigma, double nu,
			double tau, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mu, sigma, nu, tau, random);
		return result;
	}

	private final double mu;
	private final double sigma;
	private final double nu;
	private final double tau;

	public SinhArcsinh(double mu, double sigma, double nu, double tau) {
		this.mu = mu;
		this.sigma = sigma;
		this.nu = nu;
		this.tau = tau;
	}

	@Override public double density(double x, boolean log) {
		return density(x, mu, sigma, nu, tau, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, sigma, nu, tau, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu, sigma, nu, tau, lowerTail, logP);
	}
	@Override public double random() { return random(mu, sigma, nu, tau, random); }
}
