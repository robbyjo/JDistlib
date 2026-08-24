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
 * Triangular distribution with minimum {@code a}, maximum {@code b}, and
 * mode {@code c}.
 *
 * <p>The parameterization follows the GPL-2
 * <a href="https://cran.r-project.org/package=extraDistr">extraDistr</a>
 * {@code dtriang} API: {@code a < b} and {@code a <= c <= b}. The mode may
 * coincide with either endpoint.</p>
 *
 * @author Roby Joehanes
 */
public class Triangular extends GenericDistribution {
	private static boolean invalid(double a, double b, double c) {
		return Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(c)
				|| Double.isInfinite(a) || Double.isInfinite(b) || Double.isInfinite(c)
				|| !(a < b) || c < a || c > b;
	}

	private static double logOneMinusExp(double x) {
		return x > -Math.log(2.0) ? log(-Math.expm1(x)) : log1p(-exp(x));
	}

	public static double density(double x, double a, double b, double c,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)
				|| Double.isNaN(c)) return x + a + b + c;
		if (invalid(a, b, c)) return Double.NaN;
		if (x < a || x > b) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;

		double logDensity;
		if (x == c) {
			logDensity = log(2.0) - log(b - a);
		} else if (x < c) {
			logDensity = log(2.0) + log(x - a) - log(b - a) - log(c - a);
		} else {
			logDensity = log(2.0) + log(b - x) - log(b - a) - log(b - c);
		}
		return giveLog ? logDensity : exp(logDensity);
	}

	public static double cumulative(double x, double a, double b, double c,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)
				|| Double.isNaN(c)) return x + a + b + c;
		if (invalid(a, b, c)) return Double.NaN;
		if (x <= a) return boundary(false, lowerTail, logP);
		if (x >= b) return boundary(true, lowerTail, logP);

		double logF;
		double logS;
		if (x <= c) {
			logF = 2.0 * log(x - a) - log(b - a) - log(c - a);
			logS = logOneMinusExp(logF);
		} else {
			logS = 2.0 * log(b - x) - log(b - a) - log(b - c);
			logF = logOneMinusExp(logS);
		}
		double result = lowerTail ? logF : logS;
		return logP ? result : exp(result);
	}

	public static double quantile(double p, double a, double b, double c,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(a) || Double.isNaN(b)
				|| Double.isNaN(c)) return p + a + b + c;
		if (invalid(a, b, c) || invalidProbability(p, logP)) return Double.NaN;

		double logF;
		double logS;
		if (lowerTail) {
			logF = logP ? p : log(p);
			logS = logP ? logOneMinusExp(p) : log1p(-p);
		} else {
			logS = logP ? p : log(p);
			logF = logP ? logOneMinusExp(p) : log1p(-p);
		}

		double fractionAtMode = (c - a) / (b - a);
		if (logF < log(fractionAtMode)) {
			return a + exp(0.5 * (logF + log(b - a) + log(c - a)));
		}
		return b - exp(0.5 * (logS + log(b - a) + log(b - c)));
	}

	public static double random(double a, double b, double c, RandomEngine random) {
		if (invalid(a, b, c)) return Double.NaN;
		return quantile(random.nextDouble(), a, b, c, true, false);
	}

	public static double[] random(int n, double a, double b, double c,
			RandomEngine random) {
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = random(a, b, c, random);
		return values;
	}

	private static boolean invalidProbability(double p, boolean logP) {
		return logP ? p > 0.0 : p < 0.0 || p > 1.0;
	}

	private static double boundary(boolean upper, boolean lowerTail, boolean logP) {
		boolean one = upper == lowerTail;
		return logP ? (one ? 0.0 : Double.NEGATIVE_INFINITY) : (one ? 1.0 : 0.0);
	}

	private final double a;
	private final double b;
	private final double c;

	public Triangular(double a, double b, double c) {
		this.a = a;
		this.b = b;
		this.c = c;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, a, b, c, log);
	}

	@Override
	public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, a, b, c, lowerTail, logP);
	}

	@Override
	public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, a, b, c, lowerTail, logP);
	}

	@Override
	public double random() {
		return random(a, b, c, random);
	}
}
