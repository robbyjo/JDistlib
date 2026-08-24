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
import static java.lang.Math.expm1;
import static java.lang.Math.log;
import static java.lang.Math.log1p;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Gompertz distribution with shape and rate parameters.
 *
 * <p>This follows the unrestricted-shape parameterization used by the
 * GPL-2-or-later
 * <a href="https://cran.r-project.org/package=flexsurv">flexsurv</a> package.
 * A zero shape is exponential with the supplied rate. A negative shape gives a
 * defective distribution whose remaining probability is concentrated at
 * positive infinity; quantiles and random draws return positive infinity for
 * that component.</p>
 *
 * @author Roby Joehanes
 */
public class Gompertz extends GenericDistribution {
	private static boolean invalid(double shape, double rate) {
		return Double.isNaN(shape) || Double.isInfinite(shape)
				|| !(rate > 0.0) || Double.isInfinite(rate);
	}

	private static double logOneMinusExp(double x) {
		return x > -log(2.0) ? log(-expm1(x)) : log1p(-exp(x));
	}

	private static double cumulativeHazardRaw(double x, double shape,
			double rate) {
		if (shape == 0.0) return rate * x;
		double scaled = shape * x;
		if (scaled == Double.POSITIVE_INFINITY) return Double.POSITIVE_INFINITY;
		if (scaled == Double.NEGATIVE_INFINITY) return -rate / shape;
		if (scaled < -0.5) return rate / shape * expm1(scaled);
		return rate * x * exprel(scaled);
	}

	private static double exprel(double x) {
		return x == 0.0 ? 1.0 : expm1(x) / x;
	}

	public static double density(double x, double shape, double rate,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(rate)) {
			return x + shape + rate;
		}
		if (invalid(shape, rate)) return Double.NaN;
		if (x < 0.0 || x == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double scaled = shape * x;
		if (scaled == Double.POSITIVE_INFINITY) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = log(rate) + scaled - cumulativeHazardRaw(x, shape, rate);
		return giveLog ? result : exp(result);
	}

	public static double cumulative(double x, double shape, double rate,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(rate)) {
			return x + shape + rate;
		}
		if (invalid(shape, rate)) return Double.NaN;
		if (x < 0.0) return boundary(false, lowerTail, logP);

		double logSurvival = -cumulativeHazardRaw(x, shape, rate);
		double result = lowerTail ? logOneMinusExp(logSurvival) : logSurvival;
		return logP ? result : exp(result);
	}

	public static double quantile(double p, double shape, double rate,
			boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(shape) || Double.isNaN(rate)) {
			return p + shape + rate;
		}
		if (invalid(shape, rate) || invalidProbability(p, logP)) return Double.NaN;

		double cumulativeHazard;
		if (lowerTail) {
			double logSurvival = logP ? logOneMinusExp(p) : log1p(-p);
			cumulativeHazard = -logSurvival;
		} else {
			cumulativeHazard = -(logP ? p : log(p));
		}
		if (shape == 0.0) return cumulativeHazard / rate;

		double argument = shape * cumulativeHazard / rate;
		if (argument <= -1.0) return Double.POSITIVE_INFINITY;
		return log1p(argument) / shape;
	}

	public static double random(double shape, double rate, RandomEngine random) {
		if (invalid(shape, rate)) return Double.NaN;
		return quantile(random.nextDouble(), shape, rate, true, false);
	}

	public static double[] random(int n, double shape, double rate,
			RandomEngine random) {
		double[] values = new double[n];
		for (int i = 0; i < n; i++) values[i] = random(shape, rate, random);
		return values;
	}

	/** Returns the hazard {@code rate * exp(shape * x)}. */
	public static double hazard(double x, double shape, double rate,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(rate)) {
			return x + shape + rate;
		}
		if (invalid(shape, rate)) return Double.NaN;
		if (x < 0.0) return Double.NaN;
		double result = log(rate) + shape * x;
		return giveLog ? result : exp(result);
	}

	/** Returns the cumulative hazard. */
	public static double cumulativeHazard(double x, double shape, double rate,
			boolean giveLog) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(rate)) {
			return x + shape + rate;
		}
		if (invalid(shape, rate) || x < 0.0) return Double.NaN;
		double result = cumulativeHazardRaw(x, shape, rate);
		return giveLog ? log(result) : result;
	}

	private static boolean invalidProbability(double p, boolean logP) {
		return logP ? p > 0.0 : p < 0.0 || p > 1.0;
	}

	private static double boundary(boolean upper, boolean lowerTail, boolean logP) {
		boolean one = upper == lowerTail;
		return logP ? (one ? 0.0 : Double.NEGATIVE_INFINITY) : (one ? 1.0 : 0.0);
	}

	private final double shape;
	private final double rate;

	public Gompertz(double shape, double rate) {
		this.shape = shape;
		this.rate = rate;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, shape, rate, log);
	}

	@Override
	public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, shape, rate, lowerTail, logP);
	}

	@Override
	public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, shape, rate, lowerTail, logP);
	}

	@Override
	public double random() {
		return random(shape, rate, random);
	}
}
