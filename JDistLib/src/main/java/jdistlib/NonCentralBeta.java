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
import jdistlib.rng.RandomEngine;

public class NonCentralBeta extends GenericDistribution {
	public static final double density(double x, double a, double b, double ncp, boolean give_log) {
		final double eps = 1.e-15;

		int kMax;
		double k, ncp2, dx2, d, D;
		double sum, term, p_k, q;

		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp)) return x + a + b + ncp;
		if (ncp < 0 || a <= 0 || b <= 0) return Double.NaN;

		if (MathFunctions.isInfinite(a) || MathFunctions.isInfinite(b) || MathFunctions.isInfinite(ncp))
			return Double.NaN;

		if (x < 0 || x > 1) return(give_log ? Double.NEGATIVE_INFINITY : 0.);
		if(ncp == 0)
			return Beta.density(x, a, b, give_log);

		/* New algorithm, starting with *largest* term : */
		ncp2 = 0.5 * ncp;
		dx2 = ncp2*x;
		d = (dx2 - a - 1)/2;
		D = d*d + dx2 * (a + b) - a;
		if(D <= 0) {
			kMax = 0;
		} else {
			D = ceil(d + sqrt(D));
			kMax = (D > 0) ? (int)D : 0;
		}

		/* The starting "middle term" --- first look at it's log scale: */
		term = Beta.density(x, a + kMax, b, /* log = */ true);
		p_k = Poisson.density_raw(kMax, ncp2,true);
		if(x == 0. || MathFunctions.isInfinite(term) || MathFunctions.isInfinite(p_k)) /* if term = +Inf */
			//return R_D_exp(p_k + term);
			return (give_log ? (p_k + term) : exp(p_k + term));

		/* Now if s_k := p_k * t_k  {here = exp(p_k + term)} would underflow,
		 * we should rather scale everything and re-scale at the end:*/

		p_k += term; /* = log(p_k) + log(t_k) == log(s_k) -- used at end to rescale */
		/* mid = 1 = the rescaled value, instead of  mid = exp(p_k); */

		/* Now sum from the inside out */
		sum = term = 1. /* = mid term */;
		/* middle to the left */
		k = kMax;
		while(k > 0 && term > sum * eps) {
			k--;
			q = /* 1 / r_k = */ (k+1)*(k+a) / (k+a+b) / dx2;
			term *= q;
			sum += term;
		}
		/* middle to the right */
		term = 1.;
		k = kMax;
		do {
			q = /* r_{old k} = */ dx2 * (k+a+b) / (k+a) / (k+1);
			k++;
			term *= q;
			sum += term;
		} while (term > sum * eps);

		//return R_D_exp(p_k + log(sum));
		return (give_log ? (p_k + log(sum)) : exp(p_k + log(sum)));
	}

	public static final double cumulative_raw (double x, double o_x, double a, double b, double ncp) {
		/* o_x  == 1 - x  but maybe more accurate */

		/* change errmax and itrmax if desired;
		 * original (AS 226, R84) had  (errmax; itrmax) = (1e-6; 100) */
		final double errmax = 1.0e-9;
		final int    itrmax = 10000;  /* 100 is not enough for pf(ncp=200)
					     see PR#11277 */

		double a0, lbeta, c, errbd, temp, x0; //, tmp_c;

		/*long*/ double ans, ax, gx, q, sumq; // TODO long double

		if (ncp < 0. || a <= 0. || b <= 0.) return Double.NaN;

		if(x < 0. || o_x > 1. || (x == 0. && o_x == 1.)) return 0.;
		if(x > 1. || o_x < 0. || (x == 1. && o_x == 0.)) return 1.;

		c = ncp / 2.;

		/* initialize the series */

		x0 = floor(max(c - 7. * sqrt(c), 0.));
		a0 = a + x0;
		lbeta = lgammafn(a0) + lgammafn(b) - lgammafn(a0 + b);
		/* temp = pbeta_raw(x, a0, b, TRUE, FALSE), but using (x, o_x): */
		double[] tt = bratio(a0, b, x, o_x, false);
		temp = tt[0]; // tmp_c = tt[1]; ierr = (int) tt[2];

		gx = exp(a0 * log(x) + b * (x < .5 ? log1p(-x) : log(o_x))
				- lbeta - log(a0));
		if (a0 > a)
			q = exp(-c + x0 * log(c) - lgammafn(x0 + 1.));
		else
			q = exp(-c);

		sumq = 1. - q;
		ans = ax = q * temp;

		/* recurse over subsequent terms until convergence is achieved */
		double j = floor(x0); // x0 could be billions, and is in package EnvStats
		do {
			j++;
			temp -= gx;
			gx *= x * (a + b + j - 1.) / (a + j);
			q *= c / j;
			sumq -= q;
			ax = temp * q;
			ans += ax;
			errbd = (temp - gx) * sumq;
		}
		while (errbd > errmax && j < itrmax + x0);

		if (errbd > errmax) {
			// ML_ERROR(ME_PRECISION, "pnbeta");
			System.err.println("Precision error NonCentralBeta.cumulative");
		}
		if (j >= itrmax + x0) {
			//ML_ERROR(ME_NOCONV, "pnbeta");
			System.err.println("Non-convergence error NonCentralBeta.cumulative");
		}

		return ans;
	}

	static final double pnbeta2(double x, double o_x, double a, double b, double ncp, boolean lower_tail, boolean log_p)
	/* o_x  == 1 - x  but maybe more accurate */
	{
		/* long */ double ans= cumulative_raw(x, o_x, a,b, ncp); // TODO long double

		/* return R_DT_val(ans), but we want to warn about cancellation here */
		if(lower_tail) return log_p	? log(ans) : ans;
		if(ans > 1 - 1e-10) {
			// ML_ERROR(ME_PRECISION, "pnbeta");
			System.err.println("Precision error NonCentralBeta.cumulative");
		}
		ans = min(ans, 1.0);  /* Precaution */
		return log_p ? log1p(-ans) : (1 - ans);
	}

	public static final double cumulative(double x, double a, double b, double ncp, boolean lower_tail, boolean log_p) {
	    if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp)) return x + a + b + ncp;
	    // R_P_bounds_01(x, 0., 1.);
	    if(x <= 0) return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
	    if(x >= 1) return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));

	    return pnbeta2(x, 1-x, a, b, ncp, lower_tail, log_p);
	}

	public static final double quantile(double p, double a, double b, double ncp, boolean lower_tail, boolean log_p) {
		final double accu = 1e-15;
		final double Eps = 1e-14; /* must be > accu */

		double ux, lx, nx, pp;

		if (Double.isNaN(p) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp))
			return p + a + b + ncp;
		if (MathFunctions.isInfinite(a)) return Double.NaN;

		if (ncp < 0. || a <= 0. || b <= 0.) return Double.NaN;

		// R_Q_P01_boundaries(p, 0, 1);
		if (log_p) {
			if(p > 0)
				return Double.NaN;
			if(p == 0) /* upper bound*/
				return lower_tail ? 1 : 0;
			if(p == Double.NEGATIVE_INFINITY)
				return lower_tail ? 0 : 1;
		}
		else { /* !log_p */
			if(p < 0 || p > 1)
				return Double.NaN;
			if(p == 0)
				return lower_tail ? 0 : 1;
			if(p == 1)
				return lower_tail ? 1 : 0;
		}
		//p = R_DT_qIv(p);
		p = (log_p ? (lower_tail ? exp(p) : - expm1(p)) : (lower_tail ? (p) : (0.5 - (p) + 0.5)));

		/* Invert pnbeta(.) :
		 * 1. finding an upper and lower bound */
		if(p > 1 - DBL_EPSILON) return 1.0;
		pp = min(1 - DBL_EPSILON, p * (1 + Eps));
		for(ux = 0.5; ux < 1 - DBL_EPSILON && cumulative(ux, a, b, ncp, true, false) < pp; ux = 0.5*(1+ux)) ;
		pp = p * (1 - Eps);
		for(lx = 0.5; lx > DBL_MIN && cumulative(lx, a, b, ncp, true, false) > pp; lx *= 0.5) ;

		/* 2. interval (lx,ux)  halving : */
		do {
			nx = 0.5 * (lx + ux);
			if (cumulative(nx, a, b, ncp, true, false) > p) ux = nx; else lx = nx;
		} while ((ux - lx) / nx > accu);

		return 0.5 * (ux + lx);
	}

	public static final double random(double a, double b, double ncp, RandomEngine random) {
		if (ncp == 0)
			return Beta.random(a, b, random);
		double x = NonCentralChiSquare.random(2 * a, ncp, random);
		x = x / (x + NonCentralChiSquare.random(2 * b, ncp, random));
		return x;
	}

	public static final double[] random(int n, double a, double b, double ncp, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, ncp, random);
		return rand;
	}

	protected double a, b, ncp;

	public NonCentralBeta(double a, double b, double ncp) {
		this.a = a; this.b = b; this.ncp = ncp;
	}
	@Override
	public double density(double x, boolean log) {
		return density(x, a, b, ncp, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, a, b, ncp, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, a, b, ncp, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(a, b, ncp, random);
	}

	// This is taken from Baharev and Kemeny
	/*
	2-moment central F approximation; Patnaik P. B. 1949.
	The non-central chi-square and F distribution and their applications;
	Biometrika; 36: 202-232.
	*/
	static final double patnaik2(double x, double nu1, double nu2, double lambda) {
		return F.cumulative( x/(1+lambda/nu1), (nu1+lambda)*((nu1+lambda)/(nu1+2*lambda)), nu2, true, false);
	}

	/**
	This function gives an initial value of lambda for the Newton iteration.
	First, the lambda value is bracketed, then bisection is used to find a 
	better approximation. This function uses the 2-moment central F 
	approximation of Patnaik.
	 */
	static final double guess(double prob, double y, double nu1, double nu2) {
		double x, lambdal, lambdam, lambdau, fl, fm, fu;
		int itr_cnt;

		/* FIXME: cancellation ? */
		x = nu2*y/(nu1*(1.0-y));
		lambdal = 0.0;
		lambdau = 1.0;
		fl = F.cumulative(x, nu1, nu2, true, false);
		/* In this case there is no solution */
		if (fl < prob)
			throw new RuntimeException("no solution (most likely a bug)");

		fu = patnaik2(x, nu1, nu2, lambdau);
		/* Bracketing lambda: lambdal <= lambda <= lambdau */
		for (itr_cnt=1; ((fl-prob)*(fu-prob)>0.0)&&itr_cnt<=17; ++itr_cnt) {
			fl = fu;
			lambdal = lambdau;
			lambdau = 2.0*lambdau;
			fu = patnaik2(x,nu1,nu2,lambdau);
		}

		if (itr_cnt == 18)
			throw new RuntimeException("failed to bracket lambda, it is likely to be LARGE");

		/* find a better approximation of lambda by bisection */
		lambdam = (lambdal + lambdau)/2.0;
		for (itr_cnt=1; (((lambdau-lambdal)>1.0e-4*lambdau)&&((lambdau-lambdal) > 0.001))&&(itr_cnt<=29); ++itr_cnt) {
			fm = patnaik2(x, nu1, nu2, lambdam);
			if ((fm-prob)*(fu-prob) < 0.0) {
				fl = fm; lambdal = lambdam;
			} else {
				fu = fm; lambdau = lambdam;
			}
			lambdam = (lambdal + lambdau)/2.0;
		}
		if (itr_cnt == 30)
			throw new RuntimeException("failed to find initial guess");
		return lambdam;
	}

	/**
	Given prob, x, a and b, this function returns the corresponding 
	noncentrality parameter of the noncentral beta distribution.

	I.e. the following equation

	I_x(a, b, lambda) = prob

	is solved for lambda with Newton iteration.

	This function works just fine when supplied with meaningful input
	data (and from practically meaningful range) but may easily crash
	if not. Please be nice.
	*/
	public static final double calculate_ncp(double prob, double x, double a, double b) {
		double ql, qu, c, d, p, lambda, lambda_new, k, f, g, mu, eps, eps2;
		int itr_cnt;
		lambda_new = guess(prob, x, 2.0*a, 2.0*b);

		/* FIXME: are these tolerances OK ?  */
		eps  = DBL_EPSILON; //1.0e-7;
		eps2 = DBL_EPSILON; //1.0e-6;

		itr_cnt = 0;

		do {
			lambda = lambda_new;
			mu = lambda/2.0;
			ql = Poisson.quantile(eps, mu, true, false);
			qu = Poisson.quantile(eps, mu, false, false);
			k = qu;
			c = Beta.cumulative(x, a+k, b, true, false);
			d = x*(1.0-x)/(a+k-1.0)*Beta.density(x, a+k-1, b, false);
			p = Poisson.density(k, mu, false);
			f=p*c;
			p = k/mu*p;
			g = p*d;
			for (k = qu-1; k >= ql; --k) {
				c=c+d;
				d=(a+k)/(x*(a+k+b-1))*d;
				f=f+p*c;
				p=k/mu*p;
				g=g+p*d;
			}
			/* Newton step */
			lambda_new = lambda+2.0*(f-prob)/g;
			if (lambda_new <= 0.0)
				lambda_new = lambda/2.0;
			++itr_cnt;
		}
		while ((abs(lambda_new-lambda) > eps2*lambda_new)&&(itr_cnt<=10));
		if (itr_cnt == 11)
			throw new RuntimeException("newton iteration failed");
		return lambda_new;
	}
}
