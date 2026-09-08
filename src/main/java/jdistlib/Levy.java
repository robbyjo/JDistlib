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

import static java.lang.Math.*;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/**
 * 
 * @author Roby Joehanes
 *
 */
public class Levy extends GenericDistribution {
    private static boolean invalid(double mu, double sigma) {
        return !Double.isFinite(mu) || !(sigma > 0.0) || !Double.isFinite(sigma);
    }
    public static final double density(double x, double mu, double sigma, boolean giveLog) {
        if (invalid(mu, sigma) || Double.isNaN(x)) return Double.NaN;
        if (x <= mu || x == Double.POSITIVE_INFINITY) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
        double logX = log(x - mu);
        double ld = 0.5 * (log(sigma) - log(2 * PI)) - 1.5 * logX - 0.5 * sigma / (x - mu);
        return giveLog ? ld : exp(ld);
    }
    public static final double cumulative_standard(double x) { return cumulative(x, 0, 1, true, false); }
    public static final double cumulative(double x, double mu, double sigma) { return cumulative(x, mu, sigma, true, false); }
    public static final double cumulative(double x, double mu, double sigma, boolean lowerTail, boolean logP) {
        if (invalid(mu, sigma) || Double.isNaN(x)) return Double.NaN;
        if (x <= mu) return DistributionUtil.boundary(false, lowerTail, logP);
        // sigma/(X-mu) is chi-square(1); this also evaluates small survival tails directly.
        return ChiSquare.cumulative(sigma / (x - mu), 1.0, !lowerTail, logP);
    }
    public static final double quantile(double p, double mu, double sigma, boolean lowerTail, boolean logP) {
        if (invalid(mu, sigma)) return Double.NaN;
        return mu + sigma / ChiSquare.quantile(p, 1.0, !lowerTail, logP);
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
