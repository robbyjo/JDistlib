/*
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 2006 The R Core Team
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
 *  https://www.R-project.org/Licenses/
 */

#include "nmath.h"
#include "dpq.h"

/* Same AS 310 approximation used by pnbeta.c. */
static double qnbeta_crossover(double a, double b, double ncp)
{
    double half_ncp = ncp / 2.;
    double scale = fmax2(fmax2(a, b), fmax2(half_ncp, 1.));
    double as = a / scale, bs = b / scale, ls = half_ncp / scale;
    double cs = as + bs + ls;
    double correction = ls / (cs * cs * scale);
    return fmax2(0., fmin2(1., 1. - (bs / cs) * (1. + correction)));
}

double qnbeta(double p, double a, double b, double ncp,
	      int lower_tail, int log_p)
{
    double target_log, value, guess, low, high;
    int tail_lower, need_right;

#ifdef IEEE_754
    if (ISNAN(p) || ISNAN(a) || ISNAN(b) || ISNAN(ncp))
	return p + a + b + ncp;
#endif
    if (!R_FINITE(a)) ML_WARN_return_NAN;
    if (ncp < 0. || a <= 0. || b <= 0.) ML_WARN_return_NAN;

    R_Q_P01_boundaries(p, 0, 1);

    target_log = log_p ? p : log(p);
    tail_lower = lower_tail;
    /* Always invert the smaller tail.  This keeps probabilities below DBL_MIN
       representable and gives a well-conditioned comparison near one. */
    if (target_log > -M_LN2) {
	target_log = R_Log1_Exp(target_log);
	tail_lower = !tail_lower;
    }

    guess = qnbeta_crossover(a, b, ncp);
    if (!(guess > 0. && guess < 1.)) guess = .5;
    value = pnbeta(guess, a, b, ncp, tail_lower, TRUE);
    need_right = tail_lower ? value < target_log : value > target_log;

    if (need_right) {
	low = guess;
	high = low + (1. - low) * .5;
	for (int i = 0; i < 1075; i++) {
	    value = pnbeta(high, a, b, ncp, tail_lower, TRUE);
	    need_right = tail_lower ? value < target_log : value > target_log;
	    if (!need_right || high == 1.) break;
	    low = high;
	    high = low + (1. - low) * .5;
	}
    } else {
	high = guess;
	low = high * .5;
	for (int i = 0; i < 1075; i++) {
	    value = pnbeta(low, a, b, ncp, tail_lower, TRUE);
	    need_right = tail_lower ? value < target_log : value > target_log;
	    if (need_right || low == 0.) break;
	    high = low;
	    low *= .5;
	}
    }

    for (int i = 0; i < 1075; i++) {
	double mid = low == 0. ? high * .5
	    : high == 1. ? low + (1. - low) * .5
	    : low + (high - low) * .5;
	if (mid == low || mid == high) break;
	value = pnbeta(mid, a, b, ncp, tail_lower, TRUE);
	need_right = tail_lower ? value < target_log : value > target_log;
	if (need_right) low = mid; else high = mid;
    }

    return low == 0. ? high * .5 : low + (high - low) * .5;
}
