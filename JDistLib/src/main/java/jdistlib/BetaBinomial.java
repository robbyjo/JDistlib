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

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.rint;
import static jdistlib.math.MathFunctions.lgammafn;
import static jdistlib.math.MathFunctions.isInfinite;
import static jdistlib.math.MathFunctions.isNonInt;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Beta binomial distribution, taken from gamlss.dist package, plus some underflow guard.
 * Parameterization: mu, sigma, bd.
 *
 */
public class BetaBinomial extends GenericDistribution {
	/**
	 * Density
	 * @param x MUST be an integer!
	 * @param mu MUST be between 0 and 1
	 * @param sigma MUST be > 0
	 * @param bd MUST be an integer!
	 * @param give_log
	 * @return density
	 */
	public static final double density(double x, double mu, double sigma, double bd, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma) || Double.isNaN(bd)) return x + mu + sigma + bd;
		if (x < 0 || mu < 0 || mu > 1 || sigma <= 0 || bd < x) return Double.NaN;
	    if(isNonInt(x) || isNonInt(bd))
	    	return (give_log ? Double.NEGATIVE_INFINITY : 0.);
	    x = rint(x);
		double
			mu_sigma = mu / sigma,
			mu_sigma_comp = (1.0 - mu) / sigma,
			sigma_rec = 1.0 / sigma;
		double logfy = (lgammafn(bd+1) - lgammafn(x+1) - lgammafn(bd-x+1) + lgammafn(sigma_rec) + lgammafn(x+mu_sigma) +
			lgammafn(bd+mu_sigma_comp-x) - lgammafn(mu_sigma) - lgammafn(mu_sigma_comp) - lgammafn(bd+sigma_rec));
		if (isInfinite(logfy)) logfy = Binomial.density(x, bd, mu, give_log);
		return give_log ? logfy : exp(logfy);
	}

	/**
	 * Cumulative. Computed by manual summation. SLOW!
	 * @param q MUST be an integer!
	 * @param mu MUST be between 0 and 1
	 * @param sigma MUST be > 0
	 * @param bd MUST be an integer!
	 * @param lower_tail
	 * @param log_p
	 * @return cumulative
	 */
	public static final double cumulative(double q, double mu, double sigma, double bd, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(q) || Double.isNaN(mu) || Double.isNaN(sigma) || Double.isNaN(bd)) return q + mu + sigma + bd;
		if (q < 0 || mu < 0 || mu > 1 || sigma <= 0 || bd < q) return Double.NaN;
	    if(isNonInt(q) || isNonInt(bd))
	    	return (log_p ? Double.NEGATIVE_INFINITY : 0.);
	    q = rint(q);
	    double sum = 0;
	    for (int i = 0; i <= q; i++)
	    	sum += density(i, mu, sigma, bd, false);
	    if (!lower_tail) sum = 1-sum;
	    if (log_p) sum = log(sum);
	    if (isInfinite(sum)) sum = Binomial.cumulative(q, bd, mu, lower_tail, log_p);
		return sum;
	}

	/**
	 * Quantile. Computed by manual density check. SLOW!
	 * @param p MUST be between 0 and 1
	 * @param mu MUST be between 0 and 1
	 * @param sigma MUST be > 0
	 * @param bd MUST be an integer!
	 * @param lower_tail
	 * @param log_p
	 * @return quantile
	 */
	public static final double quantile(double p, double mu, double sigma, double bd, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p) || Double.isNaN(mu) || Double.isNaN(sigma) || Double.isNaN(bd)) return p + mu + sigma + bd;
		if (p < 0 || p > 1 || mu < 0 || mu > 1 || sigma <= 0 || bd < 0) return Double.NaN;
	    if(isNonInt(bd))
	    	return (log_p ? Double.NEGATIVE_INFINITY : 0.);
		if (log_p) p = exp(p);
		if (!lower_tail) p = 1-p;
		double sum = 0;
		for (int j = 0; j <= bd; j++) {
			sum += density(j, bd, mu, sigma, false);
			if (p <= sum)
				return j;
		}
		return Double.NaN;
	}

	/**
	 * Random variate
	 * @param mu MUST be between 0 and 1
	 * @param sigma MUST be > 0
	 * @param bd MUST be an integer!
	 * @param random
	 * @return random variate
	 */
	public static final double random(double mu, double sigma, double bd, RandomEngine random) {
		if (Double.isNaN(mu) || Double.isNaN(sigma) || Double.isNaN(bd)) return mu + sigma + bd;
		if (mu < 0 || mu > 1 || sigma <= 0 || bd < 0) return Double.NaN;
		double p = random.nextDouble();
		for (int j = 0; j <= bd; j++) {
			if (p <= cumulative(j, bd, mu, sigma, true, false))
				return j;
		}
		return Double.NaN;
	}

	public static final double[] random(int n, double mu, double sigma, double bd, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, sigma, bd, random);
		return rand;
	}

	protected double mu, sigma;
	protected int bd;

	public BetaBinomial(double mu, double sigma, int bd) {
		this.mu = mu; this.sigma = sigma; this.bd = bd;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, sigma, bd, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, sigma, bd, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, sigma, bd, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, sigma, bd, random);
	}
}
