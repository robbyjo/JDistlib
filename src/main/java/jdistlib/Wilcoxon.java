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
import static jdistlib.math.Constants.*;
import static jdistlib.math.MathFunctions.*;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;

/**<pre>
  SYNOPSIS

    #include <Rmath.h>
    double dwilcox(double x, double m, double n, int give_log)
    double pwilcox(double x, double m, double n, int lower_tail, int log_p)
    double qwilcox(double x, double m, double n, int lower_tail, int log_p);
    double rwilcox(double m, double n)

  DESCRIPTION

    dwilcox	The density of the Wilcoxon distribution.
    pwilcox	The distribution function of the Wilcoxon distribution.
    qwilcox	The quantile function of the Wilcoxon distribution.
    rwilcox	Random variates from the Wilcoxon distribution. </pre>
 * 
 * <P>NOTE: Since the computation of Wilcoxon distribution is highly dependent on
 * the matrix <tt>w</tt>, the dimensions of which depends on m and n (parameters of the 
 * Wilcoxon distribution), I decided to make this class as dynamic. -- Roby Joehanes
 *  
 */
public class Wilcoxon extends GenericDistribution {
	protected final int m;
	protected final int n;
	protected final double[] w;
	protected final int[] sigma;
	private int maxK = -1;

	public Wilcoxon(int m, int n) {
	    if (m < 0 || n < 0)
		throw new IllegalArgumentException("m and n must be nonnegative");
	    int i;
	    if (m > n) {
		i = n; n = m; m = i;
	    }
	    long product = (long) m * n;
	    if (product / 2 + 1 > Integer.MAX_VALUE)
		throw new IllegalArgumentException("m*n is too large for the Wilcoxon cache");
	    this.m = m;
	    this.n = n;
	    w = new double[(int) (product / 2 + 1)];
	    sigma = new int[w.length];
	}

	public int getM()
	{	return m; }

	public int getN()
	{	return n; }

	private int sigma(int k) {
		int s = 0;
		int iter1 = min(m, k);
		int iter2 = min(m + n, k);
		for (int d = 1; d <= iter1; d++)
			if (k % d == 0) s += d;
		for (int d = n + 1; d <= iter2; d++)
			if (k % d == 0) s -= d;
		return s;
	}

	private void fillTo(int newK) {
		if (newK <= maxK) return;
		for (int i = maxK + 1; i <= newK; i++)
			sigma[i] = sigma(i);
		for (int k = maxK + 1; k <= newK; k++) {
			if (k == 0) {
				w[0] = 1.;
			} else {
				double s = 0.;
				for (int i = 0; i < k; i++)
					s += w[i] * sigma[k - i];
				w[k] = s / k;
			}
		}
		maxK = newK;
	}

	protected double count(int k, int m, int n) {
		if (!((m == this.m && n == this.n) || (m == this.n && n == this.m)))
			throw new IllegalArgumentException("count parameters must match this distribution");
		int u = m * n;
		if (k < 0 || k > u)
			return(0);
		int c = u / 2;
		if (k > c)
			k = u - k; /* hence  k <= floor(u / 2) */
		if (m == 0 || n == 0 || k == 0)
			return k == 0 ? 1. : 0.;
		fillTo(k);
		return w[k];
	}

	public double density(int x, boolean give_log) {
		int m = this.m, n = this.n;
	    double d;

	    /* NaNs propagated correctly */
	    if (Double.isNaN(x) || Double.isNaN(m) || Double.isNaN(n)) return(x + m + n);
	    //m = floor(m + 0.5);
	    //n = floor(n + 0.5);
	    if (m <= 0 || n <= 0) return Double.NaN;

	    if ((x < 0) || (x > m * n))
		return (give_log ? Double.NEGATIVE_INFINITY : 0.);

	    // w_init_maybe(m, n);
	    d = give_log ?
		log(count(x, m, n)) - lchoose((double) m + n, n) :
			count(x, m, n)  /	 choose((double) m + n, n);
	    return(d);
	}

	public double cumulative(int q, boolean lower_tail, boolean log_p) {
		int m = this.m, n = this.n;
		int i;
		double c, p;

		if (Double.isNaN(q) || Double.isNaN(m) || Double.isNaN(n)) return(q + m + n);
		if (MathFunctions.isInfinite(m) || MathFunctions.isInfinite(n)) return Double.NaN;
		//m = floor(m + 0.5);
		//n = floor(n + 0.5);
		if (m <= 0 || n <= 0) return Double.NaN;

		// q = floor(q + 1e-7);

		if (q < 0.0) return(lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		if (q >= m * n) return(lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));

		//w_init_maybe(m, n);
		c = choose((double) m + n, n);
		p = 0;
		/* Use summation of probs over the shorter range */
		if (q <= (m * n / 2)) {
			for (i = 0; i <= q; i++)
				p += count(i, m, n) / c;
		}
		else {
			q = m * n - q;
			for (i = 0; i < q; i++)
				p += count(i, m, n) / c;
			lower_tail = !lower_tail; /* p = 1 - p; */
		}
		//return(R_DT_val(p));
		return (lower_tail ? (log_p ? log(p) : (p)) : (log_p ? log1p(-(p)) : (0.5 - (p) + 0.5)));
	}

	public double quantile(double x, boolean lower_tail, boolean log_p) {
		int m = this.m, n = this.n, q;
		double c, p;

		if (Double.isNaN(x) || Double.isNaN(m) || Double.isNaN(n)) return(x + m + n);
		if(MathFunctions.isInfinite(x) || MathFunctions.isInfinite(m) || MathFunctions.isInfinite(n)) return Double.NaN;
		//R_Q_P01_check(x);
		if ((log_p && x > 0) || (!log_p && (x < 0 || x > 1)) ) return Double.NaN;

		//m = floor(m + 0.5);
		//n = floor(n + 0.5);
		if (m <= 0 || n <= 0) return Double.NaN;

		if (x == (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.))) return(0);
		if (x == (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.)))
			return(m * n);

		if(log_p || !lower_tail)
			//x = R_DT_qIv(x); /* lower_tail,non-log "p" */
			x = (log_p ? (lower_tail ? exp(x) : - expm1(x)) : (lower_tail ? (x) : (0.5 - (x) + 0.5)));

		//w_init_maybe(m, n);
		c = choose((double) m + n, n);
		p = 0;
		q = 0;
		if (x <= 0.5) {
			x = x - 10 * DBL_EPSILON;
			for (;;) {
				p += count(q, m, n) / c;
				if (p >= x)
					break;
				q++;
			}
		}
		else {
			x = 1 - x + 10 * DBL_EPSILON;
			for (;;) {
				p += count(q, m, n) / c;
				if (p > x) {
					q = m * n - q;
					break;
				}
				q++;
			}
		}
		return(q);
	}

	public double random() {
		int m = this.m, n = this.n;
		int i, j, k, x[];
		double r;

		/* NaNs propagated correctly */
		if (Double.isNaN(m) || Double.isNaN(n)) return(m + n);
		//m = floor(m + 0.5);
		//n = floor(n + 0.5);
		if ((m < 0) || (n < 0))
			return Double.NaN;

		if ((m == 0) || (n == 0))
			return(0);

		r = 0.0;
		k = (int) (m + n);
		x = new int[k]; // (int *) calloc((size_t) k, sizeof(int));
		for (i = 0; i < k; i++)
			x[i] = i;
		for (i = 0; i < n; i++) {
			j = (int) floor(k * random.nextDouble());
			r += x[j];
			x[j] = x[--k];
		}
		x = null; //free(x);
		return(r - n * (n - 1) / 2);
	}

	@Override
	public double density(double x, boolean log) {
		if ((abs((x) - rint(x)) > 1e-7)) return 0;
		return density((int) x, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative((int) p, lower_tail, log_p);
	}
}
