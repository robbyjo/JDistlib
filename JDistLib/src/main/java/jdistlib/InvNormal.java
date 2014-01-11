/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998   Ross Ihaka
 *  Copyright (C) 2000-9 The R Development Core Team
 *
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
import jdistlib.math.Optimization;
import jdistlib.math.UnivariateFunction;
import jdistlib.rng.RandomEngine;
import static java.lang.Math.*;
import static jdistlib.math.Constants.DBL_MIN;

/**
 * Inverse normal (or Wald) distribution. Taken from package gamlss.dist.
 * Parameterization: mu and sigma. Note: lambda = 1/sigma^2.
 *
 */
public class InvNormal extends GenericDistribution {
	public static final double density(double x, double mu, double sigma, boolean give_log) {
		if (Double.isNaN(mu) || Double.isNaN(sigma)) return mu + sigma;
		if (mu <= 0 || sigma <= 0) return Double.NaN;
		double v = (x/mu - 1) / sigma;
		x = -0.5*log(2*PI)-log(sigma)-(3/2)*log(x)-(v*v)/(2*x);
	    return give_log ? x : exp(x);
	}

	public static final double cumulative(double q, double mu, double sigma, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(mu) || Double.isNaN(sigma)) return mu + sigma;
		if (mu <= 0 || sigma <= 0) return Double.NaN;
		double cdf1 = Normal.cumulative(((q/mu)-1)/(sigma*sqrt(q)), 0, 1, true, false);
		double lcdf2 = (2/(mu*sigma*sigma)) + Normal.cumulative((-((q/mu)+1))/(sigma*sqrt(q)), 0, 1, true, true);
		q = cdf1 + exp(lcdf2);
		if (!lower_tail) q = 1 - q; 
	    return log_p ? log(q) : q;
	}

	public static final double quantile(double p, double mu, double sigma, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(mu) || Double.isNaN(sigma)) return mu + sigma;
		if (mu <= 0 || sigma <= 0 || p < 0 || p > 1) return Double.NaN;
		if (log_p) p = exp(p);
		if (!lower_tail) p = 1-p;
		double ax = DBL_MIN, bx = mu;
		if (cumulative(mu, mu, sigma, true, false) < p) {
			ax = mu;
			int j = 1;
			do {
				bx = mu + j * sigma;
				if (cumulative(bx, mu, sigma, true, false) >= p) break;
				j++;
			} while (true);
		}
		UnivariateFunction f = new UnivariateFunction() {
			double mu, sigma, p;
			public double eval(double x) {
				return cumulative(x, mu, sigma, true, false) - p;
			}
			public void setObjects(Object... obj) {}
			public void setParameters(double... params) {
				mu = params[0];
				sigma = params[1];
				p = params[2];
			}
		};
		f.setParameters(mu, sigma, p);
		return Optimization.zeroin(f, ax, bx, 0, 10000);
	}

	public static final double random(double mu, double sigma, RandomEngine random) {
		if (Double.isNaN(mu) || Double.isNaN(sigma)) return mu + sigma;
		if (mu <= 0 || sigma <= 0) return Double.NaN;
		return quantile(random.nextDouble(), mu, sigma, true, false);
	}

	public static final double[] random(int n, double mu, double sigma, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, sigma, random);
		return rand;
	}

	protected double mu, sigma;

	public InvNormal(double mu, double sigma) {
		this.mu = mu; this.sigma = sigma;
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

	/*
	public static final void main(String[] args) {
		double x, y, z;
		for (int i = 1; i <= 600; i++) {
			x = i/100.0;
			y = cumulative(x, 1, 0.5, true, false);
			z = quantile(y, 1, 0.5, true, false);
			System.out.println(x + ", " + y + ", " + z);
		}
	}
	//*/
}
