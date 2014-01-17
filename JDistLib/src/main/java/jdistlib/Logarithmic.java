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
import jdistlib.rng.RandomEngine;

/**
 * @author Roby Joehanes
 */
public class Logarithmic extends GenericDistribution {
	public static final double density(double x, double mu, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(mu))
			return x + mu;
		if (mu <= 0 || mu >= 1)
			return Double.NaN;
		double logfy = x * log(mu) - log(x) -log(-log(1-mu));
		return give_log ? logfy : exp(logfy);
	}

	public static final double cumulative(double q, double mu, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(q) || Double.isNaN(mu))
			return q + mu;
		if (mu <= 0 || mu >= 1 || q <= 0)
			return Double.NaN;
		double sum = 0;
		for (int i = 1; i <= q; i++)
			sum += exp(i * log(mu) - log(i) -log(-log(1-mu)));
		sum = lower_tail ? sum : 1 - sum;
		return log_p ? log(sum) : sum;
	}

	public static final double quantile(double p, double mu, boolean lower_tail, boolean log_p) {
		return quantile(p, mu, lower_tail, log_p, 10000);
	}

	public static final double quantile(double p, double mu, boolean lower_tail, boolean log_p, int max_value) {
		if (Double.isNaN(p) || Double.isNaN(mu))
			return p + mu;
		if (mu <= 0 || mu >= 1)
			return Double.NaN;
		if (p < 0)
			return Double.NEGATIVE_INFINITY;
		if (p > 1)
			return Double.POSITIVE_INFINITY;
		p = log_p ? exp(p) : p;
		p = lower_tail ? p : 1 - p;
		double sum = 0;
		for (int i = 0; i < max_value; i++) {
			sum += exp(i * log(mu) - log(i) -log(-log(1-mu)));
			if (p <= sum)
				return i;
		}
		return Double.NaN;
	}

	public static final double random(double mu, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, mu, true, false);
		return u1;
	}

	public static final double[] random(int n, double mu, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, random);
		return rand;
	}

	protected double mu;

	public Logarithmic(double mu) {
		this.mu = mu;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, random);
	}
}
