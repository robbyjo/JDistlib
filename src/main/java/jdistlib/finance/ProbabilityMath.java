/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Log-probability arithmetic shared by exact distribution wrappers. */
final class ProbabilityMath {
	private ProbabilityMath() {}
	static double complement(double logP) {
		return logP < -Math.log(2.0) ? Math.log1p(-Math.exp(logP)) : Math.log(-Math.expm1(logP));
	}
	static double subtract(double a, double b) {
		if (a == b) return Double.NEGATIVE_INFINITY;
		return a + complement(b - a);
	}
	static double add(double a, double b) {
		double maximum = Math.max(a, b);
		return Double.isInfinite(maximum) ? maximum : maximum + Math.log1p(Math.exp(Math.min(a,b)-maximum));
	}
	static double logProbability(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || (logP ? p > 0.0 : p < 0.0 || p > 1.0)) return Double.NaN;
		double result = logP ? p : Math.log(p);
		return lowerTail ? result : complement(result);
	}
}
