/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Mean/size negative binomial conditional on a positive count. */
public class ZeroTruncatedNegativeBinomial extends GenericDistribution {
	public static double density(double x, double mu, double size,
			boolean giveLog) {
		return ModifiedCount.density(x, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.ZERO_TRUNCATED, mu, size, 0.0, giveLog);
	}
	public static double cumulative(double x, double mu, double size,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.cumulative(x, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.ZERO_TRUNCATED, mu, size, 0.0, lowerTail, logP);
	}
	public static double quantile(double p, double mu, double size,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.quantile(p, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.ZERO_TRUNCATED, mu, size, 0.0, lowerTail, logP);
	}
	public static double random(double mu, double size, RandomEngine random) {
		return ModifiedCount.random(ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.ZERO_TRUNCATED, mu, size, 0.0, random);
	}
	public static double[] random(int n, double mu, double size,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mu, size, random);
		return result;
	}

	private final double mu;
	private final double size;
	public ZeroTruncatedNegativeBinomial(double mu, double size) {
		this.mu = mu;
		this.size = size;
	}
	@Override public double density(double x, boolean log) {
		return density(x, mu, size, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, size, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu, size, lowerTail, logP);
	}
	@Override public double random() { return random(mu, size, random); }
}
