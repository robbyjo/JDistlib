/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Hurdle negative binomial with {@code pi} denoting positive-count mass. */
public class HurdleNegativeBinomial extends GenericDistribution {
	public static double density(double x, double mu, double size, double pi,
			boolean giveLog) {
		return ModifiedCount.density(x, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.HURDLE, mu, size, pi, giveLog);
	}
	public static double cumulative(double x, double mu, double size, double pi,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.cumulative(x, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.HURDLE, mu, size, pi, lowerTail, logP);
	}
	public static double quantile(double p, double mu, double size, double pi,
			boolean lowerTail, boolean logP) {
		return ModifiedCount.quantile(p, ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.HURDLE, mu, size, pi, lowerTail, logP);
	}
	public static double random(double mu, double size, double pi,
			RandomEngine random) {
		return ModifiedCount.random(ModifiedCount.Family.NEGATIVE_BINOMIAL,
				ModifiedCount.Kind.HURDLE, mu, size, pi, random);
	}
	public static double[] random(int n, double mu, double size, double pi,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(mu, size, pi, random);
		return result;
	}

	private final double mu;
	private final double size;
	private final double pi;
	public HurdleNegativeBinomial(double mu, double size, double pi) {
		this.mu = mu;
		this.size = size;
		this.pi = pi;
	}
	@Override public double density(double x, boolean log) {
		return density(x, mu, size, pi, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, size, pi, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu, size, pi, lowerTail, logP);
	}
	@Override public double random() { return random(mu, size, pi, random); }
}
