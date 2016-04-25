/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

import static java.lang.Math.*;

/**
 * 
 * @author Roby Joehanes
 *
 */
public class Levy extends GenericDistribution {
	public static final double density(double x, double mu, double sigma, boolean give_log) {
		double ld = 0.5 * log(sigma / (2*PI)) - 1.5 * log(x - mu) - 0.5 * sigma / (x - mu);
		return give_log ? ld : exp(ld);
	}

	public static final double cumulative_standard(double x) {
		return cumulative(x, 0, 1, true, false);
	}

	public static final double cumulative(double x, double mu, double sigma) {
		return cumulative(x, mu, sigma, true, false);
	}

	public static final double cumulative(double x, double mu, double sigma, boolean lower_tail, boolean log_p) {
		double val = Normal.cumulative(sqrt(sigma /(x - mu)), 0, 1, !lower_tail, log_p);
		return log_p? log(2) + val : 2*val;
	}

	public static final double quantile(double p, double mu, double sigma, boolean lower_tail, boolean log_p) {
		if (log_p) {
			if (MathFunctions.isInfinite(p)) return Double.POSITIVE_INFINITY;
			if (p == 0) return Double.NaN;
		} else {
			if (p < 0 || p > 1) return Double.NaN;
			if (p == 0) return Double.POSITIVE_INFINITY;
		}
		double val = Normal.quantile(log_p ? p - log(2) : p/2, 0, 1, !lower_tail, log_p) / sqrt(2);
		return mu + 0.5 * sigma / (val * val);
	}

	/**
	 * Random by quantile inversion -- the default in R
	 * @param mu
	 * @param sigma
	 * @param random
	 * @return random variate
	 */
	public static final double random(double mu, double sigma, RandomEngine random) {
		return mu + sigma * random_standard(random);
	}

	public static final double random_standard(RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, 0, 1, true, false);
		return u1;
	}

	public static final double[] random(int n, double mu, double sigma, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, sigma, random);
		return rand;
	}

	public double mu, sigma;

	/**
	 * Constructor for standard normal (i.e., mean = 0, sd = 1)
	 */
	public Levy() {
		this(0, 1);
	}

	public Levy(double mu, double sigma) {
		this.mu = mu; this.sigma = sigma;
		if (sigma <= 0) throw new RuntimeException("Sigma must be positive");
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, sigma, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, sigma, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, sigma, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, sigma, random);
	}
}
