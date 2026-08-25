/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Poisson distribution conditional on a positive count. */
public class ZeroTruncatedPoisson extends GenericDistribution {
	public static double density(double x, double lambda, boolean giveLog) {
		return ModifiedCount.density(x, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_TRUNCATED, lambda, 0.0, 0.0, giveLog);
	}
	public static double cumulative(double x, double lambda, boolean lowerTail,
			boolean logP) {
		return ModifiedCount.cumulative(x, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_TRUNCATED, lambda, 0.0, 0.0, lowerTail, logP);
	}
	public static double quantile(double p, double lambda, boolean lowerTail,
			boolean logP) {
		return ModifiedCount.quantile(p, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_TRUNCATED, lambda, 0.0, 0.0, lowerTail, logP);
	}
	public static double random(double lambda, RandomEngine random) {
		return ModifiedCount.random(ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_TRUNCATED, lambda, 0.0, 0.0, random);
	}
	public static double[] random(int n, double lambda, RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(lambda, random);
		return result;
	}

	private final double lambda;
	public ZeroTruncatedPoisson(double lambda) { this.lambda = lambda; }
	@Override public double density(double x, boolean log) {
		return density(x, lambda, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, lambda, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, lambda, lowerTail, logP);
	}
	@Override public double random() { return random(lambda, random); }
}
