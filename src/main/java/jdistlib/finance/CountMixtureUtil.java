/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;

final class CountMixtureUtil {
	private CountMixtureUtil() {}
	static int index(double value) {
		if (value >= Integer.MAX_VALUE)
			throw new IllegalArgumentException("compound-count evaluation requires x < Integer.MAX_VALUE");
		return (int) value;
	}
	static double endpoint(boolean upper, boolean lowerTail, boolean logP) {
		double result = upper == lowerTail ? 0.0 : Double.NEGATIVE_INFINITY;
		return logP ? result : Math.exp(result);
	}
	static double quantile(GenericDistribution law, double p, boolean lowerTail, boolean logP) {
		double logLower = ProbabilityMath.logProbability(p, lowerTail, logP);
		if (Double.isNaN(logLower)) return Double.NaN;
		if (logLower == Double.NEGATIVE_INFINITY) return 0.0;
		if (logLower == 0.0) return Double.POSITIVE_INFINITY;
		boolean useLower = logLower < -Math.log(2.0);
		double target = useLower ? logLower : ProbabilityMath.complement(logLower);
		int low = -1, high = 1;
		while (below(law, high, target, useLower)) {
			low = high;
			if (high > (Integer.MAX_VALUE-2)/2)
				throw new IllegalArgumentException("compound-count quantile exceeds indexed evaluation range");
			high = 2*high+1;
		}
		while (high-low > 1) {
			int middle = low + (high-low)/2;
			if (below(law, middle, target, useLower)) low=middle; else high=middle;
		}
		return high;
	}
	private static boolean below(GenericDistribution law, int x, double target, boolean lower) {
		double value = law.cumulative(x, lower, true);
		return lower ? value < target : value > target;
	}
}
