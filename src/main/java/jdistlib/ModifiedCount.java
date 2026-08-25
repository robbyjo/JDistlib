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

import jdistlib.rng.RandomEngine;

/** Shared implementation for modified Poisson and negative-binomial laws. */
final class ModifiedCount {
	enum Family { POISSON, NEGATIVE_BINOMIAL }
	enum Kind { ZERO_INFLATED, ZERO_TRUNCATED, HURDLE }

	private ModifiedCount() {}

	private static boolean invalid(Family family, double first, double second,
			double mixture, Kind kind) {
		if (Double.isNaN(first) || Double.isNaN(second)
				|| Double.isNaN(mixture)) return true;
		boolean badFamily = family == Family.POISSON
				? first < 0.0 || Double.isInfinite(first)
				: first < 0.0 || !(second > 0.0)
						|| Double.isInfinite(first) || Double.isInfinite(second);
		return badFamily || (kind != Kind.ZERO_TRUNCATED
				&& (mixture < 0.0 || mixture > 1.0));
	}

	private static double baseDensity(double x, Family family, double first,
			double second, boolean logP) {
		return family == Family.POISSON
				? Poisson.density(x, first, logP)
				: NegBinomial.density_mu(x, second, first, logP);
	}

	private static double baseCumulative(double x, Family family, double first,
			double second, boolean lowerTail, boolean logP) {
		return family == Family.POISSON
				? Poisson.cumulative(x, first, lowerTail, logP)
				: NegBinomial.cumulative_mu(x, second, first, lowerTail, logP);
	}

	static double density(double x, Family family, Kind kind, double first,
			double second, double mixture, boolean giveLog) {
		if (Double.isNaN(x)) return x;
		if (invalid(family, first, second, mixture, kind)) return Double.NaN;
		if (x < 0.0 || x != Math.rint(x) || Double.isInfinite(x)) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (kind == Kind.ZERO_INFLATED) {
			double logMass = log1p(-mixture)
					+ baseDensity(x, family, first, second, true);
			if (x == 0.0) logMass = DistributionUtil.logAdd(logMass, log(mixture));
			return giveLog ? logMass : exp(logMass);
		}
		if (first == 0.0) {
			double mass;
			if (kind == Kind.ZERO_TRUNCATED) mass = x == 1.0 ? 1.0 : 0.0;
			else mass = x == 0.0 ? 1.0 - mixture : (x == 1.0 ? mixture : 0.0);
			return giveLog ? log(mass) : mass;
		}
		if (kind == Kind.HURDLE && x == 0.0) {
			return giveLog ? log1p(-mixture) : 1.0 - mixture;
		}
		if (x < 1.0) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		double result = baseDensity(x, family, first, second, true)
				- baseCumulative(0.0, family, first, second, false, true);
		if (kind == Kind.HURDLE) result += log(mixture);
		return giveLog ? result : exp(result);
	}

	static double cumulative(double x, Family family, Kind kind, double first,
			double second, double mixture, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return x;
		if (invalid(family, first, second, mixture, kind)) return Double.NaN;
		if (x < 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		if (kind == Kind.ZERO_INFLATED) {
			double result;
			if (lowerTail) {
				result = DistributionUtil.logAdd(log(mixture), log1p(-mixture)
						+ baseCumulative(x, family, first, second, true, true));
			} else {
				result = log1p(-mixture)
						+ baseCumulative(x, family, first, second, false, true);
			}
			return logP ? result : exp(result);
		}
		if (first == 0.0) {
			double lower;
			if (kind == Kind.ZERO_TRUNCATED) lower = x < 1.0 ? 0.0 : 1.0;
			else lower = x < 1.0 ? 1.0 - mixture : 1.0;
			double value = lowerTail ? lower : 1.0 - lower;
			return logP ? log(value) : value;
		}
		if (kind == Kind.HURDLE && x < 1.0) {
			double value = lowerTail ? 1.0 - mixture : mixture;
			return logP ? log(value) : value;
		}
		if (kind == Kind.ZERO_TRUNCATED && x < 1.0) {
			return DistributionUtil.boundary(false, lowerTail, logP);
		}

		double logUpper = baseCumulative(x, family, first, second, false, true)
				- baseCumulative(0.0, family, first, second, false, true);
		if (kind == Kind.HURDLE) logUpper += log(mixture);
		double result = lowerTail ? DistributionUtil.logOneMinusExp(logUpper)
				: logUpper;
		return logP ? result : exp(result);
	}

	static double quantile(double p, Family family, Kind kind, double first,
			double second, double mixture, boolean lowerTail, boolean logP) {
		if (invalid(family, first, second, mixture, kind)) return Double.NaN;
		double minimum = kind == Kind.ZERO_TRUNCATED ? 1.0 : 0.0;
		return DistributionUtil.discreteQuantile(p, lowerTail, logP, minimum,
				Double.POSITIVE_INFINITY,
				(x, lt, lp) -> cumulative(x, family, kind, first, second,
						mixture, lt, lp));
	}

	static double random(Family family, Kind kind, double first, double second,
			double mixture, RandomEngine random) {
		return quantile(random.nextDouble(), family, kind, first, second, mixture,
				true, false);
	}
}
