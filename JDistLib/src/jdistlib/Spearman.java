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

public class Spearman {
	/**
	 * Spearman cumulative distribution. (taken from src/library/stats/src/prho.c)
	 * @param n
	 * @param is
	 * @param lower_tail
	 * @return
	 */
	public static final double cumulative(int n, double is, boolean lower_tail)
	{
		final double
		c1 = .2274,
		c2 = .2531,
		c3 = .1745,
		c4 = .0758,
		c5 = .1033,
		c6 = .3932,
		c7 = .0879,
		c8 = .0151,
		c9 = .0072,
		c10= .0831,
		c11= .0131,
		c12= 4.6e-4;
		final int n_small = 9;

		/* Local variables */
		double b, u, x, y, n3;/*, js */
		int l[] = new int[n_small];
		int nfac, i, m, mt, ifr, ise, n1;
		double pv = lower_tail ? 0. : 1.;

		if (is <= 0)
			return pv;
		n3 = n;
		n3 *= (n3 * n3 - 1.) / 3.;/* = (n^3 - n)/3 */
		if (is > n3)
			return 1 - pv;

		if (n <= n_small) { /* 2 <= n <= n_small :
		 * Exact evaluation of probability */
			nfac = 1;
			for (i = 1; i <= n; ++i) {
				nfac *= i;
				l[i - 1] = i;
			}
			/* KH mod next line: was `!=' in the code but `.eq.' in the paper */
			if (is == n3) {
				ifr = 1;
			}
			else {
				ifr = 0;
				for (m = 0; m < nfac; ++m) {
					ise = 0;
					for (i = 0; i < n; ++i) {
						n1 = i + 1 - l[i];
						ise += n1 * n1;
					}
					if (is <= ise)
						++ifr;

					n1 = n;
					do {
						mt = l[0];
						for (i = 1; i < n1; ++i)
							l[i - 1] = l[i];
						--n1;
						l[n1] = mt;
					} while (mt == n1+1 && n1 > 1);
				}
			}
			pv = (lower_tail ? nfac-ifr : ifr) / (double) nfac;
		} /* exact for n <= n_small */
		else { /* n >= n_small : Evaluation by Edgeworth series expansion */
			y = (double) (n);
			b = 1 / y;
			x = (6. * (is - 1) * b / (y * y - 1) - 1) * sqrt(y - 1);
			/* = rho * sqrt(n-1)  ==  rho / sqrt(var(rho))  ~  (0,1) */
			y = x * x;
			u = x * b * (c1 + b * (c2 + c3 * b) +
					y * (-c4 + b * (c5 + c6 * b) -
							y * b * (c7 + c8 * b -
									y * (c9 - c10 * b + y * b * (c11 - c12 * y))
									)));
			y = u / exp(y / 2.);
			pv = (lower_tail ? -y : y) + Normal.cumulative(x, 0., 1., lower_tail, /*log_p = */false);
			/* above was call to alnorm() [algorithm AS 66] */
			if (pv < 0) pv = 0.;
			if (pv > 1) pv = 1.;
		}
		return pv;
	}
}
