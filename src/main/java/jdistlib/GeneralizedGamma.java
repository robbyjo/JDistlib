/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Stacy generalized gamma distribution as parameterized by VGAM. */
public class GeneralizedGamma extends GenericDistribution {
	private static boolean invalid(double scale, double d, double k) {
		return !(scale > 0.0) || !(d > 0.0) || !(k > 0.0)
				|| Double.isInfinite(scale) || Double.isInfinite(d)
				|| Double.isInfinite(k);
	}

	public static double density(double x, double scale, double d, double k,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(d)
				|| Double.isNaN(k)) return x + scale + d + k;
		if (invalid(scale, d, k)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (x == 0.0) {
			double power = d * k;
			if (power < 1.0) return Double.POSITIVE_INFINITY;
			if (power > 1.0) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			double result = log(d) - log(scale) - lgammafn(k);
			return giveLog ? result : exp(result);
		}
		double result = log(d) - d * k * log(scale)
				+ (d * k - 1.0) * log(x) - pow(x / scale, d) - lgammafn(k);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double scale, double d, double k,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(scale) || Double.isNaN(d)
				|| Double.isNaN(k)) return x + scale + d + k;
		if (invalid(scale, d, k)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		return Gamma.cumulative(pow(x / scale, d), k, 1.0, lowerTail, logP);
	}

	public static double quantile(double p, double scale, double d, double k,
			boolean lowerTail, boolean logP) {
		if (invalid(scale, d, k)) return Double.NaN;
		double value = Gamma.quantile(p, k, 1.0, lowerTail, logP);
		return scale * pow(value, 1.0 / d);
	}

	public static double random(double scale, double d, double k,
			RandomEngine random) {
		if (invalid(scale, d, k)) return Double.NaN;
		return scale * pow(Gamma.random(k, 1.0, random), 1.0 / d);
	}
	public static double[] random(int n, double scale, double d, double k,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(scale, d, k, random);
		return result;
	}

	private final double scale;
	private final double d;
	private final double k;
	public GeneralizedGamma(double scale, double d, double k) {
		this.scale = scale; this.d = d; this.k = k;
	}
	@Override public double density(double x, boolean logP) {
		return density(x, scale, d, k, logP);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, scale, d, k, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, scale, d, k, lowerTail, logP);
	}
	@Override public double random() { return random(scale, d, k, random); }
}
