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

/** Package-private numerical helpers for contributed distributions. */
final class DistributionUtil {
	interface CumulativeFunction {
		double value(double x, boolean lowerTail, boolean logP);
	}

	private DistributionUtil() {}

	static boolean invalidProbability(double p, boolean logP) {
		return logP ? p > 0.0 : p < 0.0 || p > 1.0;
	}

	static double boundary(boolean upper, boolean lowerTail, boolean logP) {
		boolean one = upper == lowerTail;
		return logP ? (one ? 0.0 : Double.NEGATIVE_INFINITY)
				: (one ? 1.0 : 0.0);
	}

	static double logOneMinusExp(double x) {
		return x > -log(2.0) ? log(-expm1(x)) : log1p(-exp(x));
	}

	static double logAdd(double x, double y) {
		if (x == Double.NEGATIVE_INFINITY) return y;
		if (y == Double.NEGATIVE_INFINITY) return x;
		double high = Math.max(x, y);
		return high + log1p(exp(Math.min(x, y) - high));
	}

	static double discreteQuantile(double p, boolean lowerTail, boolean logP,
			double minimum, double maximum, CumulativeFunction cumulative) {
		if (Double.isNaN(p) || invalidProbability(p, logP)) return Double.NaN;
		boolean zero = p == (logP ? Double.NEGATIVE_INFINITY : 0.0);
		boolean one = p == (logP ? 0.0 : 1.0);
		if ((lowerTail && zero) || (!lowerTail && one)) return minimum;
		if ((lowerTail && one) || (!lowerTail && zero)) return maximum;

		double low = minimum;
		if (quantileCondition(low, p, lowerTail, logP, cumulative)) return low;
		double high = Math.max(low + 1.0, 1.0);
		while (high < maximum
				&& !quantileCondition(high, p, lowerTail, logP, cumulative)) {
			low = high;
			high = Math.min(maximum, high * 2.0 + 1.0);
		}
		if (high == Double.POSITIVE_INFINITY) return high;
		while (high - low > 1.0) {
			double middle = Math.floor(low + (high - low) / 2.0);
			if (quantileCondition(middle, p, lowerTail, logP, cumulative)) {
				high = middle;
			} else {
				low = middle;
			}
		}
		return high;
	}

	private static boolean quantileCondition(double x, double p,
			boolean lowerTail, boolean logP, CumulativeFunction cumulative) {
		double probability = cumulative.value(x, lowerTail, logP);
		return lowerTail ? probability >= p : probability <= p;
	}

	static double continuousQuantile(double p, boolean lowerTail, boolean logP,
			double minimum, double initialUpper, CumulativeFunction cumulative) {
		if (Double.isNaN(p) || invalidProbability(p, logP)) return Double.NaN;
		boolean zero = p == (logP ? Double.NEGATIVE_INFINITY : 0.0);
		boolean one = p == (logP ? 0.0 : 1.0);
		if ((lowerTail && zero) || (!lowerTail && one)) return minimum;
		if ((lowerTail && one) || (!lowerTail && zero)) {
			return Double.POSITIVE_INFINITY;
		}

		double low = minimum;
		double high = Math.max(initialUpper, Math.nextUp(minimum));
		while (!quantileCondition(high, p, lowerTail, logP, cumulative)) {
			low = high;
			high = high <= 0.0 ? 1.0 : high * 2.0;
			if (Double.isInfinite(high)) return high;
		}
		for (int i = 0; i < 120; i++) {
			double middle = low + (high - low) / 2.0;
			if (quantileCondition(middle, p, lowerTail, logP, cumulative)) {
				high = middle;
			} else {
				low = middle;
			}
		}
		return (low + high) / 2.0;
	}
}
