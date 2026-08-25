/*
 *  Copyright (C) 2000-2015 The R Core Team
 *
 *  The original implementation was Algorithm AS 226, with corrections
 *  AS R84 and AS R95.  This implementation uses the mode-centred,
 *  bidirectional Poisson mixture described by Benton and Krishnamoorthy
 *  (2003), together with the AS 310 approximation to the distribution mean
 *  for choosing which tail to evaluate directly.
 *
 *  Returns the cumulative probability of x for the non-central
 *  beta distribution with parameters a, b and non-centrality ncp.
 */

#include "nmath.h"
#include "dpq.h"

#define PNBETA_MAX_ITERATIONS 1000000
#define PNBETA_REANCHOR_STEPS 32

static double pnbeta_logspace_add(double x, double y)
{
    if (x == ML_NEGINF) return y;
    if (y == ML_NEGINF) return x;
    double hi = fmax2(x, y);
    return hi + log1p(exp(fmin2(x, y) - hi));
}

static double pnbeta_logspace_sub(double x, double y)
{
    if (y == ML_NEGINF) return x;
    if (!(y < x)) return ML_NAN;
    double d = y - x;
    return x + (d > -M_LN2 ? log(-expm1(d)) : log1p(-exp(d)));
}

static double pnbeta_log_beta_tail(double x, double o_x, double a, double b,
				   double index, int lower_tail)
{
    double lower, upper;
    int ierr;
    bratio(a + index, b, x, o_x, &lower, &upper, &ierr, TRUE);
    return lower_tail ? lower : upper;
}

static double pnbeta_log_beta_increment(double x, double o_x,
					double a, double b, double index)
{
    double ai = a + index;
    /* dbeta() uses Loader's deviance calculation.  Expanding the powers and
       lbeta separately loses digits when either shape is large. */
    return dbeta(x, ai, b, TRUE) + log(x) + log(o_x) - log(ai);
}

/*
 * Benton--Krishnamoorthy Method 2.  The Poisson mixture is summed in both
 * directions from its mode and entirely on the log scale.  The requested
 * incomplete-beta tail is evaluated directly, so a small upper tail is never
 * obtained by subtracting a lower-tail result from one.
 */
static double pnbeta_log_mixture_tail(double x, double o_x,
				      double a, double b, double ncp,
				      int lower_tail)
{
    const double log_eps = log(DBL_EPSILON / 2.);
    double lambda = ncp / 2.;
    if (lambda == 0.)
	return pbeta_raw(x, a, b, lower_tail, TRUE);

    double k = floor(lambda);
    /* Forward lower-tail recursion at zero subtracts two nearly equal beta
       probabilities.  For a small-mode upper tail, zero is stable and avoids
       an unnecessary backward pass. */
    if (lower_tail && k == 0.)
	k = 1.;
    else if (!lower_tail && k <= 30. && a + b > 1.)
	k = 0.;
    else if (k == 0.)
	k = 1.;

    double log_poisson = dpois_raw(k, lambda, TRUE);
    double log_tail = pnbeta_log_beta_tail(x, o_x, a, b, k, lower_tail);
    double log_increment = pnbeta_log_beta_increment(x, o_x, a, b, k);
    double log_term = log_poisson + log_tail;
    double log_sum = log_term;
    int iterations = 0;

    /* Recurse toward larger Poisson indices. */
    double i = k, previous = log_term;
    int decreasing = FALSE, direction_steps = 0;
    while (iterations++ < PNBETA_MAX_ITERATIONS) {
	double next = i + 1.;
	if (next == i) break;
	log_poisson += log(lambda) - log(next);
	double next_tail = lower_tail
	    ? pnbeta_logspace_sub(log_tail, log_increment)
	    : pnbeta_logspace_add(log_tail, log_increment);
	if (ISNAN(next_tail))
	    next_tail = pnbeta_log_beta_tail(x, o_x, a, b, next, lower_tail);
	log_increment += log(x) + log(a + b + i) - log(a + next);
	log_tail = fmin2(next_tail, 0.);
	i = next;
	if (++direction_steps % PNBETA_REANCHOR_STEPS == 0) {
	    log_poisson = dpois_raw(i, lambda, TRUE);
	    log_tail = pnbeta_log_beta_tail(x, o_x, a, b, i, lower_tail);
	    log_increment = pnbeta_log_beta_increment(x, o_x, a, b, i);
	}
	log_term = log_poisson + log_tail;
	log_sum = pnbeta_logspace_add(log_sum, log_term);
	if (log_term <= previous) decreasing = TRUE;
	if (decreasing && log_term <= log_sum + log_eps) break;
	previous = log_term;
    }

    /* Recurse from the starting index toward zero. */
    log_poisson = dpois_raw(k, lambda, TRUE);
    log_tail = pnbeta_log_beta_tail(x, o_x, a, b, k, lower_tail);
    log_increment = pnbeta_log_beta_increment(x, o_x, a, b, k);
    i = k;
    previous = log_poisson + log_tail;
    decreasing = FALSE;
    direction_steps = 0;
    while (i > 0. && iterations++ < PNBETA_MAX_ITERATIONS) {
	double next = i - 1.;
	if (next == i) break;
	log_poisson += log(i) - log(lambda);
	log_increment += log(a + i) - log(x) - log(a + b + i - 1.);
	double next_tail = lower_tail
	    ? pnbeta_logspace_add(log_tail, log_increment)
	    : pnbeta_logspace_sub(log_tail, log_increment);
	if (ISNAN(next_tail))
	    next_tail = pnbeta_log_beta_tail(x, o_x, a, b, next, lower_tail);
	log_tail = fmin2(next_tail, 0.);
	i = next;
	if (++direction_steps % PNBETA_REANCHOR_STEPS == 0) {
	    log_poisson = dpois_raw(i, lambda, TRUE);
	    log_tail = pnbeta_log_beta_tail(x, o_x, a, b, i, lower_tail);
	    log_increment = pnbeta_log_beta_increment(x, o_x, a, b, i);
	}
	log_term = log_poisson + log_tail;
	log_sum = pnbeta_logspace_add(log_sum, log_term);
	if (log_term <= previous) decreasing = TRUE;
	if (decreasing && log_term <= log_sum + log_eps) break;
	previous = log_term;
    }

    if (iterations >= PNBETA_MAX_ITERATIONS)
	ML_WARNING(ME_NOCONV, "pnbeta");
    return fmin2(log_sum, 0.);
}

/* AS 310 approximation to the non-central beta mean.  Scaling the formula
   first avoids overflow in a + b + ncp/2 and its square. */
static double pnbeta_crossover(double a, double b, double ncp)
{
    double half_ncp = ncp / 2.;
    double scale = fmax2(fmax2(a, b), fmax2(half_ncp, 1.));
    double as = a / scale, bs = b / scale, ls = half_ncp / scale;
    double cs = as + bs + ls;
    double correction = ls / (cs * cs * scale);
    return fmax2(0., fmin2(1., 1. - (bs / cs) * (1. + correction)));
}

attribute_hidden LDOUBLE
pnbeta_raw(double x, double o_x, double a, double b, double ncp)
{
    if (ncp < 0. || a <= 0. || b <= 0.) ML_WARN_return_NAN;
    if (x < 0. || o_x > 1. || (x == 0. && o_x == 1.)) return 0.;
    if (x > 1. || o_x < 0. || (x == 1. && o_x == 0.)) return 1.;
    return (LDOUBLE) exp(pnbeta_log_mixture_tail(x, o_x, a, b, ncp, TRUE));
}

attribute_hidden double
pnbeta2(double x, double o_x, double a, double b, double ncp,
	int lower_tail, int log_p)
{
    int direct_lower = x <= pnbeta_crossover(a, b, ncp);
    double direct = pnbeta_log_mixture_tail(x, o_x, a, b, ncp,
					    direct_lower);

    if (lower_tail == direct_lower)
	return log_p ? direct : exp(direct);
    return log_p ? R_Log1_Exp(direct) : -expm1(direct);
}

double pnbeta(double x, double a, double b, double ncp,
	      int lower_tail, int log_p)
{
#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(a) || ISNAN(b) || ISNAN(ncp))
	return x + a + b + ncp;
#endif

    R_P_bounds_01(x, 0., 1.);
    return pnbeta2(x, 1-x, a, b, ncp, lower_tail, log_p);
}
