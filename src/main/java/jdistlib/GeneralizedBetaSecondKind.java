/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static jdistlib.math.MathFunctions.lbeta;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Four-parameter generalized beta distribution of the second kind (GB2). */
public class GeneralizedBetaSecondKind extends GenericDistribution {
	private static boolean invalid(double scale, double a, double p, double q) {
		return !(scale > 0.0) || !(a > 0.0) || !(p > 0.0) || !(q > 0.0)
				|| Double.isInfinite(scale) || Double.isInfinite(a)
				|| Double.isInfinite(p) || Double.isInfinite(q);
	}

	private static double betaArgument(double x, double scale, double a) {
		double logPower = a * log(x / scale);
		if (logPower >= 0.0) return 1.0 / (1.0 + exp(-logPower));
		double power = exp(logPower);
		return power / (1.0 + power);
	}

	public static double density(double x, double scale, double a, double p,
			double q, boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(a)
				|| Double.isNaN(p) || Double.isNaN(q)) return x + scale + a + p + q;
		if (invalid(scale, a, p, q)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (x == 0.0) {
			double ap = a * p;
			if (ap < 1.0) return Double.POSITIVE_INFINITY;
			if (ap > 1.0) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			double result = log(a) - log(scale) - lbeta(p, q);
			return giveLog ? result : exp(result);
		}
		double logRatio = log(x / scale);
		double result = log(a) - log(scale) + (a * p - 1.0) * logRatio
				- lbeta(p, q) - (p + q) * log1p(exp(a * logRatio));
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double scale, double a, double p,
			double q, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(a)
				|| Double.isNaN(p) || Double.isNaN(q)) return x + scale + a + p + q;
		if (invalid(scale, a, p, q)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		return Beta.cumulative(betaArgument(x, scale, a), p, q, lowerTail, logP);
	}

	public static double quantile(double probability, double scale, double a,
			double p, double q, boolean lowerTail, boolean logP) {
		if (invalid(scale, a, p, q)) return Double.NaN;
		double beta = Beta.quantile(probability, p, q, lowerTail, logP);
		if (beta == 1.0) return Double.POSITIVE_INFINITY;
		return scale * exp((log(beta) - log1p(-beta)) / a);
	}

	public static double random(double scale, double a, double p, double q,
			RandomEngine random) {
		if (invalid(scale, a, p, q)) return Double.NaN;
		double beta = Beta.random(p, q, random);
		return scale * exp((log(beta) - log1p(-beta)) / a);
	}
	public static double[] random(int n, double scale, double a, double p,
			double q, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(scale, a, p, q, random);
		return result;
	}

	private final double scale, a, p, q;
	public GeneralizedBetaSecondKind(double scale, double a, double p, double q) {
		this.scale = scale; this.a = a; this.p = p; this.q = q;
	}
	@Override public double density(double x, boolean logP) {
		return density(x, scale, a, p, q, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, scale, a, p, q, lowerTail, logP);
	}
	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		return quantile(probability, scale, a, p, q, lowerTail, logP);
	}
	@Override public double random() { return random(scale, a, p, q, random); }
}
