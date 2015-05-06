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
import jdistlib.util.Bool3;

public class Beta extends GenericDistribution {

	public static final double density(double x, double a, double b, boolean log_p)
	{
	    double lval;

	    /* NaNs propagated correctly */
	    if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;

	    if (a < 0 || b < 0) return Double.NaN;
	    if (x < 0 || x > 1) return(log_p ? Double.NEGATIVE_INFINITY : 0.);

	    // limit cases for (a,b), leading to point masses
	    if(a == 0 || b == 0 || isInfinite(a) || isInfinite(b)) {
	    	if(a == 0 && b == 0) { // point mass 1/2 at each of {0,1} :
	    		if (x == 0 || x == 1) return(Double.POSITIVE_INFINITY); /* else */ return(log_p ? Double.NEGATIVE_INFINITY : 0.);
	    	}
	    	if (a == 0 || a/b == 0) { // point mass 1 at 0
	    		if (x == 0) return(Double.POSITIVE_INFINITY); /* else */ return(log_p ? Double.NEGATIVE_INFINITY : 0.);
	    	}
	    	if (b == 0 || b/a == 0) { // point mass 1 at 1
	    		if (x == 1) return(Double.POSITIVE_INFINITY); /* else */ return(log_p ? Double.NEGATIVE_INFINITY : 0.);
	    	}
	    	// else, remaining case:  a = b = Inf : point mass 1 at 1/2
	    	if (x == 0.5) return(Double.POSITIVE_INFINITY); /* else */ return(log_p ? Double.NEGATIVE_INFINITY : 0.);
	    }

	    if (x == 0) {
		if(a > 1) return(log_p ? Double.NEGATIVE_INFINITY : 0.);
		if(a < 1) return(Double.POSITIVE_INFINITY);
		/* a == 1 : */ return((log_p	? log(b) : (b)));
	    }
	    if (x == 1) {
		if(b > 1) return(log_p ? Double.NEGATIVE_INFINITY : 0.);
		if(b < 1) return(Double.POSITIVE_INFINITY);
		/* b == 1 : */ return((log_p	? log(a) : (a)));
	    }
	    if (a <= 2 || b <= 2)
		lval = (a-1)*log(x) + (b-1)*log1p(-x) - lbeta(a, b);
	    else
		lval = log(a+b-1) + Binomial.density_raw(a-1, a+b-2, x, 1-x, true);

	    return (log_p ? (lval) : exp(lval));
	}

	public static final double cumulative_raw(double x, double a, double b, boolean lower_tail, boolean log_p)
	{
	    // treat limit cases correctly here:
	    if(a == 0 || b == 0 || isInfinite(a) || isInfinite(b)) {
		// NB:  0 < x < 1 :
		if(a == 0 && b == 0) // point mass 1/2 at each of {0,1} :
		    return (log_p ? -M_LN2 : 0.5);
		if (a == 0 || a/b == 0) // point mass 1 at 0 ==> P(X <= x) = 1, all x > 0
		    return (lower_tail ? (log_p ? 0. : 1.) : log_p ? Double.NEGATIVE_INFINITY : 0.);
		if (b == 0 || b/a == 0) // point mass 1 at 1 ==> P(X <= x) = 0, all x < 1
		    return (lower_tail ? log_p ? Double.NEGATIVE_INFINITY : 0. : (log_p ? 0. : 1.));
		// else, remaining case:  a = b = Inf : point mass 1 at 1/2
		if (x < 0.5)
		    return (lower_tail ? log_p ? Double.NEGATIVE_INFINITY : 0. : (log_p ? 0. : 1.));
		// else,  x >= 0.5 :
		    return (lower_tail ? (log_p ? 0. : 1.) : log_p ? Double.NEGATIVE_INFINITY : 0.);
	    }
	    // Now:  0 < a < Inf;  0 < b < Inf
	    double x1 = 0.5 - x + 0.5, w, wc;
	    int ierr;
	    double[] temp = bratio(a, b, x, x1, log_p); /* -> ./toms708.c */
	    w = temp[0]; wc = temp[1]; ierr = (int) temp[2];
	    // ierr in {10,14} <==> bgrat() error code ierr-10 in 1:4; for 1 and 4, warned *there*
	    if(ierr > 0 && ierr != 11 && ierr != 14)
	    	//MATHLIB_WARNING(_("pbeta_raw() -> bratio() gave error code %d"), ierr);
	    	return Double.NaN;
	    return lower_tail ? w : wc;
	} /* pbeta_raw() */

	public static final double cumulative(double x, double a, double b, boolean lower_tail, boolean log_p)
	{
	    if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
	    if (a <= 0 || b <= 0) return Double.NaN;

	    if (x <= 0)	return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
	    if (x >= 1)	return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));
	    return cumulative_raw(x, a, b, lower_tail, log_p);
	}

	public static final double quantile (double alpha, double p, double q, boolean lower_tail, boolean log_p)
	{
		if (Double.isNaN(p) || Double.isNaN(q) || Double.isNaN(alpha)) return p + q + alpha;
		if (p < 0. || q < 0.) return Double.NaN;

		final double USE_LOG_X_CUTOFF = -5.;
		final int n_NEWTON_FREE = 4;

		Bool3 swap_01 = Bool3.NA;
		double log_q_cut = USE_LOG_X_CUTOFF;
		int n_N = n_NEWTON_FREE;
		double[] qb = {0, 0};

		// CARE: assumes subnormal numbers, i.e., no underflow at DBL_MIN:
		double DBL_very_MIN  = DBL_MIN / 4.,
			DBL_log_v_MIN = M_LN2*(DBL_MIN_EXP - 2),
			// Too extreme: inaccuracy in pbeta(); e.g for  qbeta(0.95, 1e-9, 20):
			// -> in pbeta() --> bgrat(..... b*z == 0 underflow, hence inaccurate pbeta()
			/* DBL_very_MIN  = 0x0.0000001p-1022, // = 2^-1050 = 2^(-1022 - 28) */
			/* DBL_log_v_MIN = -1050. * M_LN2, // = log(DBL_very_MIN) */
			// the most extreme -- not ok, as pbeta() then behaves strangely,
			// e.g., for  qbeta(0.95, 1e-8, 20):
			/* DBL_very_MIN  = 0x0.0000000000001p-1022, // = 2^-1074 = 2^(-1022 -52) */
			/* DBL_log_v_MIN = -1074. * M_LN2, // = log(DBL_very_MIN) */
			DBL_1__eps = 0x1.fffffffffffffp-1; // = 1 - 2^-53

		final double fpu = 3e-308;
		/* acu_min:  Minimal value for accuracy 'acu' which will depend on (a,p); acu_min >= fpu ! */
		final double acu_min = 1e-300;
		final double p_lo = fpu;
		final double p_hi = 1-2.22e-16;

		final double const1 = 2.30753;
		final double const2 = 0.27061;
		final double const3 = 0.99229;
		final double const4 = 0.04481;

		boolean
			swap_choose = (swap_01 == Bool3.NA),
			swap_tail,
			log_, give_log_q = (log_q_cut == Double.POSITIVE_INFINITY),
			use_log_x = give_log_q, // or u < log_q_cut  below
			warned = false, add_N_step = true,
			skip_to_return = false;
		int i_pb, i_inn;
		double a, la, logbeta, g, h, pp, p_, qq, r, s, t, w, y = -1.;
		double u = 0, u_n = 1, xinbta = 0;

		// return_q_0 == return give_log_q ? Double.NEGATIVE_INFINITY : 0;
		// return_q_1 == return give_log_q ? 0 : 1;
		// return_q_half == return give_log_q ? -M_LN2 : 0.5;

		// if(alpha == R_DT_0)
		if(alpha == (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.)))
			return give_log_q ? Double.NEGATIVE_INFINITY : 0;

		// if(alpha == R_DT_1)
		if(alpha == (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.)))
			return give_log_q ? 0 : 1;

		// check alpha {*before* transformation which may all accuracy}:
		if((log_p && alpha > 0) ||
				(!log_p && (alpha < 0 || alpha > 1))) { // alpha is outside
			//R_ifDEBUG_printf("qbeta(alpha=%g, %g, %g, .., log_p=%d): %s%s\n",
			//		alpha, p,q, log_p, "alpha not in ",
			//		log_p ? "[-Inf, 0]" : "[0,1]");
			// ML_ERR_return_NAN :
			//ML_ERROR(ME_DOMAIN, "");
			//qb[0] = qb[1] = ML_NAN; return;
			return Double.NaN;
		}

	    if (p == 0 || q == 0 || MathFunctions.isInfinite(p) || MathFunctions.isInfinite(q)) {
	    	if(p == 0 && q == 0) { // point mass 1/2 at each of {0,1} :
	    	    if(alpha < (log_p ? -M_LN2 : 0.5)) return give_log_q ? Double.NEGATIVE_INFINITY : 0;
	    	    if(alpha > (log_p ? -M_LN2 : 0.5)) return give_log_q ? 0 : 1;
	    	    // else:  alpha == "1/2"
	    	    return give_log_q ? -M_LN2 : 0.5;
	    	} else if (p == 0 || p/q == 0) { // point mass 1 at 0 - "flipped around"
	    		return give_log_q ? Double.NEGATIVE_INFINITY : 0;
	    	} else if (q == 0 || q/p == 0) { // point mass 1 at 0 - "flipped around"
	    		return give_log_q ? 0 : 1;
	    	}
	    	// else:  p = q = Inf : point mass 1 at 1/2
	    	return give_log_q ? -M_LN2 : 0.5;
	    }

		/* initialize */
	    //p_ = R_DT_qIv(alpha);/* lower_tail prob (in any case) */
	    p_ = (log_p ? (lower_tail ? exp(alpha) : - expm1(alpha)) : (lower_tail ? (alpha) : (0.5 - (alpha) + 0.5)));
	    // Conceptually,  0 < p_ < 1  (but can be 0 or 1 because of cancellation!)
		logbeta = lbeta(p, q);

		swap_tail = (swap_choose) ? (p_ > 0.5) : swap_01.v();
		// change tail; default (swap_01 = NA): afterwards 0 < a <= 1/2
		if(swap_tail) { /* change tail, swap  p <-> q :*/
			//a = R_DT_CIv(alpha); // = 1 - p_ < 1/2
			a = (log_p ? (lower_tail ? -expm1(alpha) : exp(alpha)) : (lower_tail ? (0.5 - (alpha) + 0.5) : (alpha)));
			/* la := log(a), but without numerical cancellation: */
			//la = lower_tail ? R_D_LExp(alpha) : R_D_log(alpha);
			la = lower_tail ? (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha)) : (log_p ? (alpha) : log(alpha));
			pp = q; qq = p;
		}
		else {
			a = p_;
			//la = lower_tail ? R_D_log(alpha) : R_D_LExp(alpha);
			la = lower_tail ? (log_p ? (alpha) : log(alpha)) : (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha));
			pp = p; qq = q;
		}

		/* calculate the initial approximation */

	    /* Desired accuracy for Newton iterations (below) should depend on  (a,p)
	     * This is from Remark .. on AS 109, adapted.
	     * However, it's not clear if this is "optimal" for IEEE double prec.

	     * acu = fmax2(acu_min, pow(10., -25. - 5./(pp * pp) - 1./(a * a)));

	     * NEW: 'acu' accuracy NOT for squared adjustment, but simple;
	     * ---- i.e.,  "new acu" = sqrt(old acu)
	    */
		double acu = max(acu_min, pow(10., -13. - 2.5/(pp * pp) - 0.5/(a * a)));
	    // try to catch  "extreme left tail" early
		double tx = xinbta, u0 = (la + log(pp) + logbeta) / pp; // = log(x_0)
		final double
		log_eps_c = M_LN2 * (1. - DBL_MANT_DIG);// = log(DBL_EPSILON) = -36.04..
		r = pp*(1.-qq)/(pp+1.);

		t = 0.2;
		// FIXME: Factor 0.2 is a bit arbitrary;  '1' is clearly much too much.

		//R_ifDEBUG_printf("qbeta(%g, %g, %g, lower_t=%d, log_p=%d):%s\n  swap_tail=%d, la=%g, u0=%g (bnd: %g (%g)) ",
		//		alpha, p,q, lower_tail, log_p,
		//		(log_p && (p_ == 0. || p_ == 1.)) ? (p_==0.?" p_=0":" p_=1") : "",
		//				swap_tail, la, u0,
		//				(t*log_eps_c - log(abs(pp*(1.-qq)*(2.-qq)/(2.*(pp+2.)))))/2.,
		//				t*log_eps_c - log(abs(r))
		//		);

		if(M_LN2 * DBL_MIN_EXP < u0 && // cannot allow exp(u0) = 0 ==> exp(u1) = exp(u0) = 0
				u0 < -0.01 && // (must: u0 < 0, but too close to 0 <==> x = exp(u0) = 0.99..)
				// qq <= 2 && // <--- "arbitrary"
				// u0 <  t*log_eps_c - log(fabs(r)) &&
				u0 < (t*log_eps_c - log(abs(pp*(1.-qq)*(2.-qq)/(2.*(pp+2.)))))/2.)
		{
			// TODO: maybe jump here from below, when initial u "fails" ?
			// L_tail_u:
			// MM's one-step correction (cheaper than 1 Newton!)
			r = r*exp(u0);// = r*x0
			if(r > -1.) {
				u = u0 - log1p(r)/pp;
				//R_ifDEBUG_printf("u1-u0=%9.3g --> choosing u = u1\n", u-u0);
			} else {
				u = u0;
				//R_ifDEBUG_printf("cannot cheaply improve u0\n");
			}
			tx = xinbta = exp(u);
			use_log_x = true; // or (u < log_q_cut)  ??
			//goto L_Newton;
		} else {
			// y := y_\alpha in AS 64 := Hastings(1955) approximation of qnorm(1 - a) :
			r = sqrt(-2 * la);
			y = r - (const1 + const2 * r) / (1. + (const3 + const4 * r) * r);

			if (pp > 1 && qq > 1) { // use  Carter(1947), see AS 109, remark '5.'
				r = (y * y - 3.) / 6.;
				s = 1. / (pp + pp - 1.);
				t = 1. / (qq + qq - 1.);
				h = 2. / (s + t);
				w = y * sqrt(h + r) / h - (t - s) * (r + 5. / 6. - 2. / (3. * h));
				//R_ifDEBUG_printf("p,q > 1 => w=%g", w);
				if(w > 300) { // exp(w+w) is huge or overflows
					t = w+w + log(qq) - log(pp); // = argument of log1pexp(.)
					u = // log(xinbta) = - log1p(qq/pp * exp(w+w)) = -log(1 + exp(t))
							(t <= 18) ? -log1p(exp(t)) : -t - exp(-t);
							xinbta = exp(u);
				} else {
					xinbta = pp / (pp + qq * exp(w + w));
					u = // log(xinbta)
							- log1p(qq/pp * exp(w+w));
				}
			} else { // use the original AS 64 proposal, Scheffé-Tukey (1944) and Wilson-Hilferty
				r = qq + qq;
				/* A slightly more stable version of  t := \chi^2_{alpha} of AS 64
				 * t = 1. / (9. * qq); t = r * R_pow_di(1. - t + y * sqrt(t), 3);  */
				t = 1. / (3. * sqrt(qq));
				t = r * pow(1. + t*(-t + y), 3);// = \chi^2_{alpha} of AS 64
				s = 4. * pp + r - 2.;// 4p + 2q - 2 = numerator of new t = (...) / chi^2
				//R_ifDEBUG_printf("min(p,q) <= 1: t=%g", t);
				if (t == 0 || (t < 0. && s >= t)) { // cannot use chisq approx
					// x0 = 1 - { (1-a)*q*B(p,q) } ^{1/q}    {AS 65}
					// xinbta = 1. - exp((log(1-a)+ log(qq) + logbeta) / qq);
					double l1ma;/* := log(1-a), directly from alpha (as 'la' above):
					 * FIXME: not worth it? log1p(-a) always the same ?? */
					if(swap_tail)
						//l1ma = lower_tail ? R_D_log(alpha) : R_D_LExp(alpha);
						l1ma = lower_tail ? (log_p ? (alpha) : log(alpha)) : (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha));
					else
						//l1ma = lower_tail ? R_D_LExp(alpha) : R_D_log(alpha);
						l1ma = lower_tail ? (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha)) : (log_p ? (alpha) : log(alpha));
					//R_ifDEBUG_printf(" t <= 0 : log1p(-a)=%.15g, better l1ma=%.15g\n", log1p(-a), l1ma);
					double xx = (l1ma + log(qq) + logbeta) / qq;
					if(xx <= 0.) {
						xinbta = -expm1(xx);
						//u = R_Log1_Exp (xx);// =  log(xinbta) = log(1 - exp(...A...))
						u = ((xx) > -M_LN2 ? log(-expm1(xx)) : log1p(-exp(xx)));// =  log(xinbta) = log(1 - exp(...A...))
					} else { // xx > 0 ==> 1 - e^xx < 0 .. is nonsense
						//R_ifDEBUG_printf(" xx=%g > 0: xinbta:= 1-e^xx < 0\n", xx);
						xinbta = 0; u = Double.NEGATIVE_INFINITY; /// FIXME can do better?
					}
				} else {
					t = s / t;
					//R_ifDEBUG_printf(" t > 0 or s < t < 0:  new t = %g ( > 1 ?)\n", t);
					if (t <= 1.) { // cannot use chisq, either
						u = (la + log(pp) + logbeta) / pp;
						xinbta = exp(u);
					} else { // (1+x0)/(1-x0) = t,  solved for x0 :
						xinbta = 1. - 2. / (t + 1.);
						u = log1p(-2. / (t + 1.));
					}
				}
			}

			// Problem: If initial u is completely wrong, we make a wrong decision here
			if(swap_choose &&
				(( swap_tail && u >= -exp(  log_q_cut)) || // ==> "swap back"
				(!swap_tail && u >= -exp(4*log_q_cut) && pp / qq < 1000.))) { // ==> "swap now" (much less easily)
				// "revert swap" -- and use_log_x
				swap_tail = !swap_tail;
				//R_ifDEBUG_printf(" u = %g (e^u = xinbta = %.16g) ==> ", u, xinbta);
				if(swap_tail) {
					//a = R_DT_CIv(alpha); // needed ?
					a = (log_p ? (lower_tail ? -expm1(alpha) : exp(alpha)) : (lower_tail ? (0.5 - (alpha) + 0.5) : (alpha)));
					/* la := log(a), but without numerical cancellation: */
					//la = lower_tail ? R_D_LExp(alpha) : R_D_log(alpha);
					la = lower_tail ? (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha)) : (log_p ? (alpha) : log(alpha));
					pp = q; qq = p;
				}
				else {
					a = p_;
					//la = lower_tail ? R_D_log(alpha) : R_D_LExp(alpha);
					la = lower_tail ? (log_p ? (alpha) : log(alpha)) : (log_p ? ((alpha) > -M_LN2 ? log(-expm1(alpha)) : log1p(-exp(alpha))) : log1p(-alpha));
					pp = p; qq = q;
				}
				//R_ifDEBUG_printf("\"%s\"; la = %g\n", (swap_tail ? "swap now" : "swap back"), la);
				// we could redo computations above, but this should be stable
				//u = R_Log1_Exp(u);
				u = ((u) > -M_LN2 ? log(-expm1(u)) : log1p(-exp(u)));
				xinbta = exp(u);

				/* Careful: "swap now"  should not fail if
					1) the above initial xinbta is "completely wrong"
					2) The correction step can go outside (u_n > 0 ==>  e^u > 1 is illegal)
					e.g., for qbeta(0.2066, 0.143891, 0.05)
				 */
			}

			if(!use_log_x)
				use_log_x = (u < log_q_cut);//(per default) <==> xinbta = e^u < 4.54e-5
			boolean
			bad_u = Double.isInfinite(u),
			bad_init = bad_u || xinbta > p_hi;

			//R_ifDEBUG_printf(" -> u = %g, e^u = xinbta = %.16g, (Newton acu=%g%s)\n",
			//	u, xinbta, acu, (bad_u ? ", ** bad u **" : (use_log_x ? ", on u = log(x) scale" : "")));

			u_n = 1.; // -Wall
			tx = xinbta; // keeping "original initial x" (for now)

			if(bad_u || u < log_q_cut) { /* e.g.
				    qbeta(0.21, .001, 0.05)
				    try "left border" quickly, i.e.,
				    try at smallest positive number: */
				w = Beta.cumulative_raw(DBL_very_MIN, pp, qq, true, log_p);
				if(w > (log_p ? la : a)) {
					//R_ifDEBUG_printf(" quantile is left of smallest positive number; \"convergence\"\n");
					if(log_p || abs(w - a) < abs(0 - a)) { // DBL_very_MIN is better than 0
						tx   = DBL_very_MIN;
						u_n  = DBL_log_v_MIN;// = log(DBL_very_MIN)
					} else {
						tx   = 0.;
						u_n  = Double.NEGATIVE_INFINITY;
					}
					use_log_x = log_p; add_N_step = false; //goto L_return;
					skip_to_return = true; // TODO
				}
				else {
					// R_ifDEBUG_printf(" pbeta(smallest pos.) = %g <= %g  --> continuing\n", w, (log_p ? la : a));
					if(u < DBL_log_v_MIN) {
						u = DBL_log_v_MIN;// = log(DBL_very_MIN)
						xinbta = DBL_very_MIN;
					}
				}
			}


			if (!skip_to_return) {
				/* Sometimes the approximation is negative (and == 0 is also not "ok") */
				if (bad_init && !(use_log_x && tx > 0)) {
					if(u == Double.NEGATIVE_INFINITY) {
						//R_ifDEBUG_printf("  u = -Inf;");
						u = M_LN2 * DBL_MIN_EXP;
						xinbta = DBL_MIN;
					} else {
						//R_ifDEBUG_printf(" bad_init: u=%g, xinbta=%g;", u,xinbta);
						xinbta = (xinbta > 1.1) // i.e. "way off"
								? 0.5 // otherwise, keep the respective boundary:
										: ((xinbta < p_lo) ? exp(u) : p_hi);
						if(bad_u)
							u = log(xinbta);
						// otherwise: not changing "potentially better" u than the above
					}
					//R_ifDEBUG_printf(" -> (partly)new u=%g, xinbta=%g\n", u,xinbta);
				}
			}
		}

		//L_Newton: // TODO
		if (!skip_to_return) {
			boolean converged = false;
			/* --------------------------------------------------------------------
			 * Solve for x by a modified Newton-Raphson method, using pbeta_raw()
			 */
			r = 1 - pp;
			t = 1 - qq;
			double wprev = 0., prev = 1., adj = 1.; // -Wall

			if(use_log_x) { // find  log(xinbta) -- work in  u := log(x) scale
				// if(bad_init && tx > 0) xinbta = tx;// may have been better

				for (i_pb=0; i_pb < 1000; i_pb++) {
					// using log_p == TRUE  unconditionally here
					// FIXME: if exp(u) = xinbta underflows to 0, like different formula pbeta_log(u, *)
					y = Beta.cumulative_raw(xinbta, pp, qq, /*lower_tail = */ true, true);

					/* w := Newton step size for   L(u) = log F(e^u)  =!= 0;   u := log(x)
					 *   =  (L(.) - la) / L'(.);  L'(u)= (F'(e^u) * e^u ) / F(e^u)
					 *   =  (L(.) - la)*F(.) / {F'(e^u) * e^u } =
					 *   =  (L(.) - la) * e^L(.) * e^{-log F'(e^u) - u}
					 *   =  ( y   - la) * e^{ y - u -log F'(e^u)}
			        and  -log F'(x)= -log f(x) =  + logbeta + (1-p) log(x) + (1-q) log(1-x)
				               = logbeta + (1-p) u + (1-q) log(1-e^u)
					 */
					w = (y == Double.NEGATIVE_INFINITY) // y = -Inf  well possible: we are on log scale!
						? 0. : (y - la) * exp(y - u + logbeta + r * u + t * ((u) > -M_LN2 ? log(-expm1(u)) : log1p(-exp(u))));
					if(Double.isInfinite(w))
						break;
					if (i_pb >= n_N && w * wprev <= 0.)
						prev = max(abs(adj),fpu);
					//R_ifDEBUG_printf("N(i=%2d): u=%#20.16g, pb(e^u)=%#12.6g, w=%#15.9g, %s prev=%11g,",
					//		i_pb, u, y, w, (w * wprev <= 0.) ? "new" : "old", prev);
					g = 1;
					for (i_inn=0; i_inn < 1000; i_inn++) {
						adj = g * w;
						// take full Newton steps at the beginning; only then safe guard:
						if (i_pb < n_N || abs(adj) < prev) {
							u_n = u - adj; // u_{n+1} = u_n - g*w
							if (u_n <= 0.) { // <==> 0 <  xinbta := e^u  <= 1
								if (prev <= acu || abs(w) <= acu) {
									/* R_ifDEBUG_printf(" -adj=%g, %s <= acu  ==> convergence\n", */
									/* 	 -adj, (prev <= acu) ? "prev" : "|w|"); */
									//R_ifDEBUG_printf(" it{in}=%d, -adj=%g, %s <= acu  ==> convergence\n",
									//	i_inn, -adj, (prev <= acu) ? "prev" : "|w|");
									//goto L_converged;
									converged = true; break;
								}
								// if (u_n != ML_NEGINF && u_n != 1)
								break;
							}
						}
						g /= 3;
					}
					if (converged) break;
					// (cancellation in (u_n -u) => may differ from adj:
					double D = min(abs(adj), abs(u_n - u));
					/* R_ifDEBUG_printf(" delta(u)=%g\n", u_n - u); */
					//R_ifDEBUG_printf(" it{in}=%d, delta(u)=%9.3g, D/|.|=%.3g\n", i_inn, u_n - u, D/abs(u_n + u));
					if (D <= 4e-16 * abs(u_n + u))
						//goto L_converged;
						{ converged = true; break; }
					u = u_n;
					xinbta = exp(u);
					wprev = w;
				} // for(i )

			} else

				for (i_pb=0; i_pb < 1000; i_pb++) {
					y = Beta.cumulative_raw(xinbta, pp, qq, /*lower_tail = */ true, log_p);
					// delta{y} :   d_y = y - (log_p ? la : a);
					if(Double.isInfinite(y) && !(log_p && y == Double.NEGATIVE_INFINITY))// y = -Inf  is ok if(log_p)
					{ // ML_ERR_return_NAN :
						//ML_ERROR(ME_DOMAIN, "");
						System.err.println("MEDOMAIN on Beta.quantile");
						//qb[0] = qb[1] = ML_NAN; return;
						return Double.NaN;
					}


					/* w := Newton step size  (F(.) - a) / F'(.)  or,
					 * --   log: (lF - la) / (F' / F) = exp(lF) * (lF - la) / F'
					 */
					w = log_p
						? (y - la) * exp(y + logbeta + r * log(xinbta) + t * log1p(-xinbta))
						: (y - a)  * exp(    logbeta + r * log(xinbta) + t * log1p(-xinbta));
					if (i_pb >= n_N && w * wprev <= 0.)
						prev = max(abs(adj),fpu);
					//R_ifDEBUG_printf("N(i=%2d): x0=%#17.15g, pb(x0)=%#17.15g, w=%#17.15g, %s prev=%g,",
					//	i_pb, xinbta, y, w, (w * wprev <= 0.) ? "new" : "old", prev);
					g = 1;
					for (i_inn=0; i_inn < 1000;i_inn++) {
						adj = g * w;
						// take full Newton steps at the beginning; only then safe guard:
						if (i_pb < n_N || abs(adj) < prev) {
							tx = xinbta - adj; // x_{n+1} = x_n - g*w
							if (0. <= tx && tx <= 1.) {
								if (prev <= acu || abs(w) <= acu) {
									//R_ifDEBUG_printf(" it{in}=%d, delta(x)=%g, %s <= acu  ==> convergence\n",
									//	i_inn, -adj, (prev <= acu) ? "prev" : "|w|");
									//goto L_converged;
									converged = true; break;
								}
								if (tx != 0. && tx != 1)
									break;
							}
						}
						g /= 3;
					}
					if (converged) break;
					//R_ifDEBUG_printf(" it{in}=%d, delta(x)=%g\n", i_inn, tx - xinbta);
					if (abs(tx - xinbta) <= 4e-16 * (tx + xinbta)) // "<=" : (.) == 0
						//goto L_converged;
						{ converged = true; break; }
					xinbta = tx;
					if(tx == 0) // "we have lost"
						break;
					wprev = w;
				}

			/*-- NOT converged: Iteration count --*/
			if (!converged) {
				warned = true;
				//ML_ERROR(ME_PRECISION, "qbeta");
			}

			//L_converged:
			log_ = log_p || use_log_x; // only for printing
			//R_ifDEBUG_printf(" %s: Final delta(y) = %g%s\n", warned ? "_NO_ convergence" : "converged",
			//	y - (log_ ? la : a), (log_ ? " (log_)" : ""));
			if((log_ && y == Double.NEGATIVE_INFINITY) || (!log_ && y == 0)) {
				// stuck at left, try if smallest positive number is "better"
				w = Beta.cumulative_raw(DBL_very_MIN, pp, qq, true, log_);
				if(log_ || abs(w - a) <= abs(y - a)) {
					tx  = DBL_very_MIN;
					u_n = DBL_log_v_MIN;// = log(DBL_very_MIN)
				}
				add_N_step = false; // not trying to do better anymore
			}
			else if(!warned && (log_ ? abs(y - la) > 3 : abs(y - a) > 1e-4)) {
				if(!(log_ && y == Double.NEGATIVE_INFINITY &&
						// e.g. qbeta(-1e-10, .2, .03, log=TRUE) cannot get accurate ==> do NOT warn
						Beta.cumulative_raw(DBL_1__eps, // = 1 - eps
								pp, qq, true, true) > la + 2))
					//MATHLIB_WARNING2( // low accuracy for more platform independent output:
					System.err.println(String.format(
							"qbeta(a, *) =: x0 with |pbeta(x0,*%s) - alpha| = %.5g is not accurate",
							(log_ ? ", log_" : ""), abs(y - (log_ ? la : a))));
			}
		}

		//L_return:
		if(give_log_q) { // ==> use_log_x , too
			//if(!use_log_x) // (see if claim above is true)
			//	MATHLIB_WARNING("qbeta() L_return, u_n=%g;  give_log_q=TRUE but use_log_x=FALSE -- please report!", u_n);
			//double r = R_Log1_Exp(u_n);
			r = ((u_n) > -M_LN2 ? log(-expm1(u_n)) : log1p(-exp(u_n)));
			if(swap_tail) {
				qb[0] = r;	 qb[1] = u_n;
			} else {
				qb[0] = u_n; qb[1] = r;
			}
		} else {
			if(use_log_x) {
				if(add_N_step) {
					/* add one last Newton step on original x scale, e.g., for qbeta(2^-98, 0.125, 2^-96) */
					xinbta = exp(u_n);
					y = Beta.cumulative_raw(xinbta, pp, qq, /*lower_tail = */ true, log_p);
					w = log_p
						? (y - la) * exp(y + logbeta + r * log(xinbta) + t * log1p(-xinbta))
						: (y - a)  * exp(    logbeta + r * log(xinbta) + t * log1p(-xinbta));
					tx = xinbta - w;
					//R_ifDEBUG_printf("Final Newton correction(non-log scale): xinbta=%.16g, y=%g, w=%g. => new tx=%.16g\n", xinbta, y, w, tx);
				} else {
					if(swap_tail) {
						qb[0] = -expm1(u_n); qb[1] =  exp  (u_n);
					} else {
						qb[0] =  exp  (u_n); qb[1] = -expm1(u_n);
					}
					return qb[0];
				}
			}
			if(swap_tail) {
				qb[0] = 1 - tx;	qb[1] = tx;
			} else {
				qb[0] = tx; 	qb[1] = 1 - tx;
			}
		}
		return qb[0];
	}

	public static final double random(double aa, double bb, RandomEngine random)
	{
		final double expmax = (DBL_MAX_EXP * M_LN2);
		double a, b, alpha;
		double r, s, t, u1, u2, v, w, y, z;

		//boolean qsame;
		/* FIX-ME:  Keep Globals (properly) for threading */
		/* Uses these GLOBALS to save time when many rv's are generated : */
		// RJ's modification: The paltry saving isn't worth it since we much prefer threading
		double beta, gamma, delta, k1, k2;
		//double olda = -1.0;
		//double oldb = -1.0;

		if (aa <= 0. || bb <= 0. || (isInfinite(aa) && isInfinite(bb)))
			return Double.NaN;

		if (isInfinite(aa))
			return 1.0;

		if (isInfinite(bb))
			return 0.0;

		/* Test if we need new "initializing" */
		//qsame = (olda == aa) && (oldb == bb);
		//if (!qsame) { olda = aa; oldb = bb; }

		a = min(aa, bb);
		b = max(aa, bb); /* a <= b */
		alpha = a + b;

		//#define v_w_from__u1_bet(AA)
		//	    v = beta * log(u1 / (1.0 - u1));
		//	    if (v <= expmax) {
		//		w = AA * exp(v);
		//		if(Double.isInfinite(w)) w = DBL_MAX;
		//	    } else
		//		w = DBL_MAX;


		if (a <= 1.0) {	/* --- Algorithm BC --- */

			/* changed notation, now also a <= b (was reversed) */

			//if (!qsame) { /* initialize */
			beta = 1.0 / a;
			delta = 1.0 + b - a;
			k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
			k2 = 0.25 + (0.5 + 0.25 / delta) * a;
			//}
			/* FIXME: "do { } while()", but not trivially because of "continue"s:*/
			for(;;) {
				u1 = random.nextDouble();
				u2 = random.nextDouble();
				if (u1 < 0.5) {
					y = u1 * u2;
					z = u1 * y;
					if (0.25 * u2 + z - y >= k1)
						continue;
				} else {
					z = u1 * u1 * u2;
					if (z <= 0.25) {
						//v_w_from__u1_bet(b);
						v = beta * log(u1 / (1.0 - u1));
						if (v <= expmax) {
							w = b * exp(v);
							if(isInfinite(w)) w = DBL_MAX;
						} else
							w = DBL_MAX;
						break;
					}
					if (z >= k2)
						continue;
				}

				//v_w_from__u1_bet(b);
				v = beta * log(u1 / (1.0 - u1));
				if (v <= expmax) {
					w = b * exp(v);
					if(isInfinite(w)) w = DBL_MAX;
				} else
					w = DBL_MAX;

				if (alpha * (log(alpha / (a + w)) + v) - 1.3862944 >= log(z))
					break;
			}
			return (aa == a) ? a / (a + w) : w / (a + w);

		}
		else {		/* Algorithm BB */

			//if (!qsame) { /* initialize */
			beta = sqrt((alpha - 2.0) / (2.0 * a * b - alpha));
			gamma = a + 1.0 / beta;
			//}
			do {
				u1 = random.nextDouble();
				u2 = random.nextDouble();

				//v_w_from__u1_bet(a);
				v = beta * log(u1 / (1.0 - u1));
				if (v <= expmax) {
					w = a * exp(v);
					if(isInfinite(w)) w = DBL_MAX;
				} else
					w = DBL_MAX;

				z = u1 * u1 * u2;
				r = gamma * v - 1.3862944;
				s = a + r - w;
				if (s + 2.609438 >= 5.0 * z)
					break;
				t = log(z);
				if (s > t)
					break;
			}
			while (r + alpha * log(alpha / (b + w)) < t);

			return (aa != a) ? b / (b + w) : w / (b + w);
		}
	}

	public static final double[] random(int n, double aa, double bb, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(aa, bb, random);
		return rand;
	}

	protected double a, b;

	public Beta(double a, double b) {
		this.a = a; this.b = b;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, a, b, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, a, b, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, a, b, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(a, b, random);
	}

}
