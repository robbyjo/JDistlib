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

import jdistlib.exception.PrecisionException;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;
import jdistlib.util.Debug;

public class NonCentralBeta extends GenericDistribution {
	public static final double density(double x, double a, double b, double ncp, boolean give_log) {
		final double eps = 1.e-15;

		int kMax;
		double k, ncp2, dx2, d, D;
		double sum, term, p_k, q; // #TODO Should be long double

		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp)) return x + a + b + ncp;
		if (ncp < 0 || a <= 0 || b <= 0) return Double.NaN;

		if (MathFunctions.isInfinite(a) || MathFunctions.isInfinite(b) || MathFunctions.isInfinite(ncp))
			return Double.NaN;

		if (x < 0 || x > 1) return(give_log ? Double.NEGATIVE_INFINITY : 0.);
		if(ncp == 0)
			return Beta.density(x, a, b, give_log);

		/* New algorithm, starting with *largest* term : */
		ncp2 = scalb(ncp, -1);
		dx2 = ncp2*x;
		d = scalb(dx2 - a - 1, -1);
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

	private static final int CDF_MAX_ITERATIONS = 1_000_000;
	private static final double LOG_CDF_EPSILON = log(DBL_EPSILON / 2.);

	private static double logAdd(double x, double y) {
		if (x == Double.NEGATIVE_INFINITY) return y;
		if (y == Double.NEGATIVE_INFINITY) return x;
		double hi = max(x, y);
		return hi + log1p(exp(min(x, y) - hi));
	}

	private static double logSubtract(double x, double y) {
		if (y == Double.NEGATIVE_INFINITY) return x;
		if (!(y < x)) return Double.NaN;
		double d = y - x;
		return x + (d > -M_LN2 ? log(-expm1(d)) : log1p(-exp(d)));
	}

	private static double logBetaTail(double x, double o_x, double a, double b,
			double index, boolean lowerTail) {
		double[] tails = bratio(a + index, b, x, o_x, true);
		return lowerTail ? tails[0] : tails[1];
	}

	private static double logBetaIncrement(double x, double o_x, double a,
			double b, double index) {
		double ai = a + index;
		/* Use dbeta's Loader-style deviance calculation.  Expanding the powers
		 * and lbeta separately loses digits when a or b is large. */
		return Beta.density(x, ai, b, true) + log(x) + log(o_x) - log(ai);
	}

	/*
	 * Benton--Krishnamoorthy mode-centred Poisson mixture (their Method 2,
	 * also used by Boost.Math).  Unlike AS 226, this recurses in both
	 * directions from the Poisson mode and works on the log scale.  The lower
	 * and upper incomplete-beta mixtures are evaluated directly, so a small
	 * upper tail is never formed by subtracting from one.  AS 310's crossover
	 * approximation is used below to select the smaller tail.
	 */
	private static double logMixtureTail(double x, double o_x, double a,
			double b, double ncp, boolean lowerTail) {
		double lambda = scalb(ncp, -1);
		if (lambda == 0.)
			return Beta.cumulative_raw(x, a, b, lowerTail, true);

		double k = floor(lambda);
		/* Forward lower-tail recursion from zero subtracts two nearly equal
		 * incomplete-beta values.  Starting at one avoids that cancellation.
		 * For the upper tail, starting at zero is stable and cheaper when the
		 * Poisson mode is small. */
		if (lowerTail && k == 0.)
			k = 1.;
		else if (!lowerTail && k <= 30. && a + b > 1.)
			k = 0.;
		else if (k == 0.)
			k = 1.;
		double logPoisson = Poisson.density_raw(k, lambda, true);
		double logTail = logBetaTail(x, o_x, a, b, k, lowerTail);
		double logIncrement = logBetaIncrement(x, o_x, a, b, k);
		double logTerm = logPoisson + logTail;
		double logSum = logTerm;
		int iterations = 0;

		/* Recurse toward larger Poisson indices. */
		double i = k;
		double previous = logTerm;
		boolean decreasing = false;
		int directionSteps = 0;
		while (iterations++ < CDF_MAX_ITERATIONS) {
			double next = i + 1.;
			if (next == i) break;
			logPoisson += log(lambda) - log(next);
			double nextTail = lowerTail
				? logSubtract(logTail, logIncrement)
				: logAdd(logTail, logIncrement);
			if (Double.isNaN(nextTail))
				nextTail = logBetaTail(x, o_x, a, b, next, lowerTail);
			logIncrement += log(x) + log(a + b + i) - log(a + next);
			logTail = min(nextTail, 0.);
			i = next;
			if (++directionSteps % 32 == 0) {
				logPoisson = Poisson.density_raw(i, lambda, true);
				logTail = logBetaTail(x, o_x, a, b, i, lowerTail);
				logIncrement = logBetaIncrement(x, o_x, a, b, i);
			}
			logTerm = logPoisson + logTail;
			logSum = logAdd(logSum, logTerm);
			if (logTerm <= previous) decreasing = true;
			if (decreasing && logTerm <= logSum + LOG_CDF_EPSILON) break;
			previous = logTerm;
		}

		/* Recurse toward zero. */
		logPoisson = Poisson.density_raw(k, lambda, true);
		logTail = logBetaTail(x, o_x, a, b, k, lowerTail);
		logIncrement = logBetaIncrement(x, o_x, a, b, k);
		i = k;
		previous = logPoisson + logTail;
		decreasing = false;
		directionSteps = 0;
		while (i > 0. && iterations++ < CDF_MAX_ITERATIONS) {
			double next = i - 1.;
			if (next == i) break;
			logPoisson += log(i) - log(lambda);
			logIncrement += log(a + i) - log(x) - log(a + b + i - 1.);
			double nextTail = lowerTail
				? logAdd(logTail, logIncrement)
				: logSubtract(logTail, logIncrement);
			if (Double.isNaN(nextTail))
				nextTail = logBetaTail(x, o_x, a, b, next, lowerTail);
			logTail = min(nextTail, 0.);
			i = next;
			if (++directionSteps % 32 == 0) {
				logPoisson = Poisson.density_raw(i, lambda, true);
				logTail = logBetaTail(x, o_x, a, b, i, lowerTail);
				logIncrement = logBetaIncrement(x, o_x, a, b, i);
			}
			logTerm = logPoisson + logTail;
			logSum = logAdd(logSum, logTerm);
			if (logTerm <= previous) decreasing = true;
			if (decreasing && logTerm <= logSum + LOG_CDF_EPSILON) break;
			previous = logTerm;
		}

		if (iterations >= CDF_MAX_ITERATIONS) {
			System.err.println("Non-convergence error NonCentralBeta.cumulative");
			if (Debug.warningAsError)
				throw new PrecisionException("Non-convergence error NonCentralBeta.cumulative", logSum);
		}
		return min(logSum, 0.);
	}

	public static final double cumulative_raw(double x, double o_x, double a, double b, double ncp) {
		if (ncp < 0. || a <= 0. || b <= 0.) return Double.NaN;
		if (x <= 0. || o_x >= 1.) return 0.;
		if (x >= 1. || o_x <= 0.) return 1.;
		return exp(logMixtureTail(x, o_x, a, b, ncp, true));
	}

	private static double crossover(double a, double b, double ncp) {
		double halfNcp = scalb(ncp, -1);
		double scale = max(max(a, b), max(halfNcp, 1.));
		double as = a / scale, bs = b / scale, ls = halfNcp / scale;
		double cs = as + bs + ls;
		double correction = ls / (cs * cs * scale);
		return max(0., min(1., 1. - (bs / cs) * (1. + correction)));
	}

	static final double pnbeta2(double x, double o_x, double a, double b, double ncp,
			boolean lower_tail, boolean log_p) {
		boolean directLower = x <= crossover(a, b, ncp);
		double direct = logMixtureTail(x, o_x, a, b, ncp, directLower);

		if (lower_tail == directLower)
			return log_p ? direct : exp(direct);
		if (log_p)
			return direct > -M_LN2 ? log(-expm1(direct)) : log1p(-exp(direct));
		return -expm1(direct);
	}

	public static final double cumulative(double x, double a, double b, double ncp, boolean lower_tail, boolean log_p) {
	    if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp)) return x + a + b + ncp;
	    if (!Double.isFinite(a) || !Double.isFinite(b) || !Double.isFinite(ncp)
			|| a <= 0. || b <= 0. || ncp < 0.) return Double.NaN;
	    // R_P_bounds_01(x, 0., 1.);
	    if(x <= 0) return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
	    if(x >= 1) return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));

	    return pnbeta2(x, 1-x, a, b, ncp, lower_tail, log_p);
	}

	public static final double quantile(double p, double a, double b, double ncp, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p) || Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp))
			return p + a + b + ncp;
		if (!Double.isFinite(a) || !Double.isFinite(b) || !Double.isFinite(ncp)
				|| ncp < 0. || a <= 0. || b <= 0.) return Double.NaN;

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
		double targetLog = log_p ? p : log(p);
		boolean tailLower = lower_tail;
		/* Invert the smaller tail.  This preserves log probabilities below the
		 * range of a double and makes the comparison well conditioned. */
		if (targetLog > -M_LN2) {
			targetLog = log(-expm1(targetLog));
			tailLower = !tailLower;
		}

		double guess = crossover(a, b, ncp);
		if (!(guess > 0. && guess < 1.)) guess = .5;
		double value = cumulative(guess, a, b, ncp, tailLower, true);
		boolean needRight = tailLower ? value < targetLog : value > targetLog;
		double low, high;
		if (needRight) {
			low = guess;
			high = low + (1. - low) * .5;
			for (int i = 0; i < 1075; i++) {
				value = cumulative(high, a, b, ncp, tailLower, true);
				needRight = tailLower ? value < targetLog : value > targetLog;
				if (!needRight || high == 1.) break;
				low = high;
				high = low + (1. - low) * .5;
			}
		} else {
			high = guess;
			low = high * .5;
			for (int i = 0; i < 1075; i++) {
				value = cumulative(low, a, b, ncp, tailLower, true);
				needRight = tailLower ? value < targetLog : value > targetLog;
				if (needRight || low == 0.) break;
				high = low;
				low *= .5;
			}
		}

		for (int i = 0; i < 1075; i++) {
			double mid = low == 0. ? high * .5
				: high == 1. ? low + (1. - low) * .5 : low + (high - low) * .5;
			if (mid == low || mid == high) break;
			value = cumulative(mid, a, b, ncp, tailLower, true);
			needRight = tailLower ? value < targetLog : value > targetLog;
			if (needRight) low = mid; else high = mid;
		}
		return low == 0. ? high * .5 : low + (high - low) * .5;
	}

	public static final double random(double a, double b, double ncp, RandomEngine random) {
		if (Double.isNaN(a) || Double.isNaN(b) || Double.isNaN(ncp)
				|| !Double.isFinite(a) || !Double.isFinite(b) || !Double.isFinite(ncp)
				|| a <= 0. || b <= 0. || ncp < 0.)
			return Double.NaN;
		if (ncp == 0.)
			return Beta.random(a, b, random);
		double x = NonCentralChiSquare.random(2 * a, ncp, random);
		double y = Gamma.random(b, 2., random);
		if (x > y)
			return 1. / (1. + y / x);
		double ratio = x / y;
		return ratio / (1. + ratio);
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
