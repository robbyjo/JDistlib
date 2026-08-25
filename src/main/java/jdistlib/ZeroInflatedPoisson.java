/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Poisson distribution with an additional structural-zero probability. */
public class ZeroInflatedPoisson extends GenericDistribution {
	public static double density(double x, double lambda, double pi,
			boolean giveLog) {
		return ModifiedCount.density(x, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_INFLATED, lambda, 0.0, pi, giveLog);
	}

	public static double cumulative(double x, double lambda, double pi,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.cumulative(x, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_INFLATED, lambda, 0.0, pi, lowerTail, logP);
	}

	public static double quantile(double p, double lambda, double pi,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.quantile(p, ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_INFLATED, lambda, 0.0, pi, lowerTail, logP);
	}

	public static double random(double lambda, double pi, RandomEngine random) {
		return ModifiedCount.random(ModifiedCount.Family.POISSON,
				ModifiedCount.Kind.ZERO_INFLATED, lambda, 0.0, pi, random);
	}

	public static double[] random(int n, double lambda, double pi,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(lambda, pi, random);
		return result;
	}

	private final double lambda;
	private final double pi;

	public ZeroInflatedPoisson(double lambda, double pi) {
		this.lambda = lambda;
		this.pi = pi;
	}

	@Override public double density(double x, boolean log) {
		return density(x, lambda, pi, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, lambda, pi, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, lambda, pi, lowerTail, logP);
	}
	@Override public double random() { return random(lambda, pi, random); }
}
