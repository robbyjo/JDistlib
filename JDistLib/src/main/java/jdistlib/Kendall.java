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

import static java.lang.Math.*;
import static jdistlib.math.MathFunctions.*;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/**
 * Kendall tau distribution
 *
 */
public class Kendall extends GenericDistribution {
	static final long count(int k, int n, long w[][]) {
		int i, u;
		long sum;

		u = (n * (n - 1) / 2);
		if ((k < 0) || (k > u)) return(0);
		if (w[n] == null) {
			w[n] = new long[u+1]; // (double *) R_alloc(u + 1, sizeof(double));
			//memset(w[n], '\0', sizeof(double) * (u+1));
			for (i = 0; i <= u; i++)
				w[n][i] = -1;
		}
		if (w[n][k] < 0) {
			if (n == 1)
				w[n][k] = (k == 0) ? 1 : 0;
			else {
				sum = 0;
				for (i = 0; i < n; i++)
					sum += count(k - i, n - 1, w);
				w[n][k] = sum;
			}
		}
		return(w[n][k]);
	}

	public static final double calculate_tau(double x, int n) {
		return (4 * x - 2) / (n * (n - 1)) - 1;
	}

	public static final double calculate_count(double tau, int n) {
		return 0.5 + (1.0 + tau) * n * (n - 1) / 4.0;
	}

	/**
	 * Density of Kendall distribution
	 * @param x This is count, not tau!
	 * @param n
	 * @return density
	 */
	public static final double density(double x, int n) {
		long w[][] = new long[n+1][];
		if (isNonInt(x) || x < 0 || x > (n * (n - 1) / 2)) return 0;
		return count((int) x, n, w) / gammafn(n + 1);
	}

	/**
	 * Density of Kendall distribution
	 * @param tau This is tau, not count!
	 * @param n
	 * @return density
	 */
	public static final double density_tau(double tau, int n) {
		return density(calculate_count(tau, n), n);
	}

	/**
	 * Cumulative density function of Kendall distribution.
	 * <P>Kendall statistics: x = round((rho + 1) * n * (n-1) / 4);
	 * <P>Two-sided test: min(1, 2*((q > n*(n-1)/4) ? 1-cumulative(x-1,n) : cumulative(x,n)));
	 * <P>Greater test: 1-cumulative(x-1,n)
	 * <P>Less test: cumulative(x,n)
	 * @param x This is count, not tau!
	 * @param n
	 * @return cumulative
	 */
	public static final double cumulative(double x, int n) {
		double p, q;
		long w[][] = new long[n+1][];

		q = floor(x + 1e-7);
		if (q < 0)
			return 0;
		if (q > (n * (n - 1) / 2))
			return 1;
		p = 0;
		for (int j = 0; j <= q; j++) {
			p += count(j, n, w);
		}
		return exp(log(p) - lgammafn(n + 1));
	}

	/**
	 * Cumulative distribution of Kendall distribution
	 * @param tau This is tau, not count!
	 * @param n
	 * @return cumulative
	 */
	public static final double cumulative_tau(double tau, int n) {
		return cumulative(calculate_count(tau, n), n);
	}

	/**
	 * Quantile search.
	 * @param p
	 * @param n
	 * @return count
	 */
	public static final double quantile(double p, int n) {
		if (Double.isNaN(p) || MathFunctions.isInfinite(p)) return p;
		if (p < 0 || p > 1 || n < 2) return Double.NaN;
		double mu, sigma;

		mu = n * (n-1)/4.0;
		sigma = sqrt((n * (2.0 * n + 1) * (n + 1) / 6.0 - n) / 12.0);
		long k = (long) (sigma * Normal.quantile(p, 0., 1., true, false) + mu + 0.5);

		if (p <= cumulative(k, n)) {
			do {
				k--;
				if (p > cumulative(k, n)) return k + 1.5;
			} while (k > 0);
		} else {
			do {
				k++;
				if (p <= cumulative(k, n)) return k + 0.5;
			} while (true);
		}
		return k + 0.5;
	}

	public static final double quantile_tau(double p, int n) {
		return calculate_count(quantile(p, n), n);
	}

	/**
	 * Kendall RNG by inversion
	 * @param n
	 * @param random
	 * @return random variate
	 */
	public static final double random(int n, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, n);
		return u1;
	}

	public static final double[] random(int count, int n, RandomEngine random) {
		double[] rand = new double[count];
		for (int i = 0; i < count; i++)
			rand[i] = random(n, random);
		return rand;
	}

	protected int n;

	public Kendall(int n) {
		this.n = n;
	}

	@Override
	public double density(double x, boolean log) {
		return log ? log(density(x, n)) : density(x, n);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		p = cumulative(p, n);
		return log_p ? log(lower_tail ? p : 1-p) : (lower_tail ? p : 1-p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		if (log_p) q = exp(q);
		if (!lower_tail) q = 1 - q;
		return quantile(q, n);
	}

	@Override
	public double random() {
		return random(n, random);
	}

	/*
	public static final void main(String[] args) {
		System.out.println(cumulative(26-1, 9));
		System.out.println(cumulative_tau(0.36111111111111116045, 9));
		System.out.println(quantile_tau(0.9402805335097011, 9));

		System.out.println(density_tau(0, 10));
		System.out.println(density_tau(1, 10));
		System.out.println(cumulative_tau(0, 10));
		System.out.println(cumulative_tau(0.42222222222222227650, 10));
		System.out.println(cumulative_tau(0.26315789473684203514, 20));

		System.out.println(quantile_tau(0.95, 10));
		System.out.println(quantile_tau(0.95, 20));
	}
	//*/
}
