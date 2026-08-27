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

public class NegBinomial extends GenericDistribution {
	private static final double EXTREME_MEAN_THRESHOLD = 1e50;

	public static final double density(double x, double size, double prob, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(prob)) return x + size + prob;

		if (prob <= 0 || prob > 1 || size < 0) return Double.NaN;
		//R_D_nonint_check(x);
		if(isNonInt(x)) {
			//MATHLIB_WARNING("non-integer x = %f", x);
			return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		}

		if (x < 0 || MathFunctions.isInfinite(x)) return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		//x = R_D_forceint(x);
		x = rint(x);
		if (prob == 1 || size == 0)
			return x == 0 ? (give_log ? 0. : 1.)
					: (give_log ? Double.NEGATIVE_INFINITY : 0.);
		/* With fixed prob < 1 the mass escapes to +Inf as size tends to
		 * infinity.  Treat that limit explicitly instead of replacing Inf by
		 * Double.MAX_VALUE and entering overflow-prone finite formulas. */
		if (MathFunctions.isInfinite(size))
			return give_log ? Double.NEGATIVE_INFINITY : 0.;
		if (x == 0) {
			return give_log ? size * log(prob) : pow(prob, size);
		}
		if (x < 1e-10 * size) {
			double xx2s = x < sqrt(Double.MAX_VALUE)
				? scalb(x * (x - 1), -1) / size
				: x * (scalb(x, -1) / size);
			double ans = size * log(prob) + x * (log(size) + log1p(-prob))
				- lgamma1p(x) + log1p(xx2s);
			return give_log ? ans : exp(ans);
		}
		if (hasExtremeMean(size, prob)) {
			double logDensity = extremeLogDensity(x, size, prob);
			return give_log ? logDensity : exp(logDensity);
		}
		double p = give_log
			? (x < size ? log1p(-x/(size+x)) : log(size/(size+x)))
			: size/(size+x);
		double ans = Binomial.density_raw(size, x+size, prob, 1-prob, give_log);
		return give_log ? p + ans : p * ans;
	}

	public static final double density_mu(double x, double size, double mu, boolean give_log) {
		/* originally, just set  prob :=  size / (size + mu)  and called dbinom_raw(),
		 * but that suffers from cancellation when   mu << size  */

		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(mu)) return x + size + mu;

		if (mu < 0 || size < 0) return Double.NaN;
		// R_D_nonint_check(x);
		if(isNonInt(x)) {
			//MATHLIB_WARNING("non-integer x = %f", x);
			return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		}

		if (x < 0 || MathFunctions.isInfinite(x)) return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		if (x == 0 && size==0) return (give_log ? 0. : 1.);
		if (MathFunctions.isInfinite(size))
			return Poisson.density_raw(x, mu, give_log);
		//x = R_D_forceint(x);
		x = rint(x);
		if(x == 0) { /* be accurate, both for n << mu, and n >> mu :*/
			x = size * (size < mu ? log(size/(size+mu)) : log1p(- mu/(size+mu)));
			return (give_log ? (x) : exp(x));
		}
		if(x < 1e-10 * size) { /* don't use dbinom_raw() but MM's formula: */
			/* FIXME --- 1e-8 shows problem; rather use algdiv() from ./toms708.c */
			double p = (size < mu ? log(size/(1 + size/mu)) : log(mu / (1 + mu/size)));
			double xx2s = x < sqrt(Double.MAX_VALUE)
				? scalb(x * (x - 1), -1) / size
				: x * (scalb(x, -1) / size);
			x = x * p - mu - lgamma1p(x) + log1p(xx2s);
			return (give_log ? (x) : exp(x));
		}
		/* else: no unnecessary cancellation inside dbinom_raw, when
		 * x_ = size and n_ = x+size are so close that n_ - x_ loses accuracy
		 */
		double ans = Binomial.density_raw(size, x+size, size/(size+mu), mu/(size+mu), give_log);
		double p = give_log
			? (x < size ? log1p(-x/(size+x)) : log(size/(size+x)))
			: size/(size+x);
		return give_log ? p + ans : p * ans;
	}

	public static final double cumulative(double x, double size, double prob, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(prob)) return x + size + prob;
		if(MathFunctions.isInfinite(prob)) return Double.NaN;
		if (size < 0 || prob <= 0 || prob > 1)	return Double.NaN;

		/* limiting case: point mass at zero */
		if (size == 0 || prob == 1)
			return (x >= 0) ? (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.))
				: (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		if (x < 0) return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		if (MathFunctions.isInfinite(x)) return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));
		if (MathFunctions.isInfinite(size))
			return lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.)
					: (log_p ? 0. : 1.);
		x = floor(x + 1e-7);
		if (hasExtremeMean(size, prob))
			return extremeCumulative(x, size, prob, lower_tail, log_p);
		return Beta.cumulative(prob, size, x + 1, lower_tail, log_p);
	}

	public static final double cumulative_mu(double x, double size, double mu, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(x) || Double.isNaN(size) || Double.isNaN(mu)) return x + size + mu;
		if (MathFunctions.isInfinite(mu)) return Double.NaN;
		if (size < 0 || mu < 0) return Double.NaN;

		/* limiting case: point mass at zero */
		if (size == 0)
			return (x >= 0) ? (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.))
				: (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));

			if (x < 0) return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		if (MathFunctions.isInfinite(x)) return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));
		if (MathFunctions.isInfinite(size))
			return Poisson.cumulative(x, mu, lower_tail, log_p);

		x = floor(x + 1e-7);
		/* return
		 * pbeta(pr, size, x + 1, lower_tail, log_p);  pr = size/(size + mu), 1-pr = mu/(size+mu)
		 *
		 *= pbeta_raw(pr, size, x + 1, lower_tail, log_p)
		 *            x.  pin   qin
		 *=  bratio (pin,  qin, x., 1-x., &w, &wc, &ierr, log_p),  and return w or wc ..
		 *=  bratio (size, x+1, pr, 1-pr, &w, &wc, &ierr, log_p) */
		{
			double w, wc;
			double[] temp = bratio(size, x+1, size/(size+mu), mu/(size+mu), log_p);
			w = temp[0]; wc = temp[1];
			//if(temp[2] > 0)
			//	MATHLIB_WARNING(_("pnbinom_mu() -> bratio() gave error code %d"), ierr);
			return lower_tail ? w : wc;
		}
	}

	static final double do_search(double y, double []z, double p, double n, double pr, double incr) {
		if(z[0] >= p) {
			/* search to the left */
			for(;;) {
				if(y == 0 ||
						(z[0] = cumulative(y - incr, n, pr, /*l._t.*/true, /*log_p*/false)) < p)
					return y;
				y = max(0, y - incr);
			}
		}
		else {		/* search to the right */
			for(;;) {
				y = y + incr;
				if((z[0] = cumulative(y, n, pr, /*l._t.*/true, /*log_p*/false)) >= p)
					return y;
			}
		}
	}

	public static final double quantile(double p, double size, double prob, boolean lower_tail, boolean log_p) {
		double P, Q, mu, sigma, gamma, z, y;

		if (Double.isNaN(p) || Double.isNaN(size) || Double.isNaN(prob)) return p + size + prob;
		/* this happens if specified via mu, size, since
	       prob == size/(size+mu)
		 */
		if (prob == 0 && size == 0) return 0;
	    if (prob <= 0 || prob > 1 || size < 0) return Double.NaN;
	    if (prob == 1 || size == 0) return 0;

		// R_Q_P01_boundaries(p, 0, ML_POSINF);
		if (log_p) {
			if(p > 0)
				return Double.NaN;
			if(p == 0) /* upper bound*/
				return lower_tail ? Double.POSITIVE_INFINITY : 0;
			if(p == Double.NEGATIVE_INFINITY)
				return lower_tail ? 0 : Double.POSITIVE_INFINITY;
		}
		else { /* !log_p */
			if(p < 0 || p > 1)
				return Double.NaN;
			if(p == 0)
				return lower_tail ? 0 : Double.POSITIVE_INFINITY;
			if(p == 1)
				return lower_tail ? Double.POSITIVE_INFINITY : 0;
		}
		if (MathFunctions.isInfinite(size)) return Double.POSITIVE_INFINITY;

		Q = 1.0 / prob;
		P = (1.0 - prob) * Q;
		mu = size * P;
		if (hasExtremeMean(size, prob))
			return Double.isFinite(mu) ? rint(mu) : Double.POSITIVE_INFINITY;
		sigma = sqrt(size * P * Q);
		gamma = (Q + P)/sigma;

		return DiscreteQuantile.quantile(p, lower_tail, log_p, mu, sigma, gamma,
				Double.POSITIVE_INFINITY,
				(value, lt, lp) -> cumulative(value, size, prob, lt, lp));
	}

	public static final double quantile_mu(double p, double size, double mu, boolean lower_tail, boolean log_p) {
		if (size == Double.POSITIVE_INFINITY)
			return Poisson.quantile(p, mu, lower_tail, log_p);
		if (Double.isNaN(p) || Double.isNaN(size) || Double.isNaN(mu))
			return p + size + mu;
		if (mu == 0 || size == 0)
			return 0;
		if (mu < 0 || size < 0)
			return Double.NaN;
		if (log_p) {
			if (p > 0) return Double.NaN;
			if (p == 0) return lower_tail ? Double.POSITIVE_INFINITY : 0;
			if (p == Double.NEGATIVE_INFINITY) return lower_tail ? 0 : Double.POSITIVE_INFINITY;
		} else {
			if (p < 0 || p > 1) return Double.NaN;
			if (p == 0) return lower_tail ? 0 : Double.POSITIVE_INFINITY;
			if (p == 1) return lower_tail ? Double.POSITIVE_INFINITY : 0;
		}

		double P = mu / size;
		double Q = 1 + P;
		double sigmaMu = sqrt(size * P * Q);
		double gammaMu = (Q + P) / sigmaMu;
		return DiscreteQuantile.quantile(p, lower_tail, log_p, mu, sigmaMu, gammaMu,
				Double.POSITIVE_INFINITY,
				(value, lt, lp) -> cumulative_mu(value, size, mu, lt, lp));
	}

	public static final double random(double size, double prob, RandomEngine random) {
		if(MathFunctions.isInfinite(prob) || Double.isNaN(size) || size <= 0 || prob <= 0 || prob > 1)
			/* prob = 1 is ok, PR#1218 */
			return Double.NaN;
		if (prob == 1) return 0;
		if (MathFunctions.isInfinite(size)) return Double.POSITIVE_INFINITY;
		return Poisson.random(Gamma.random(size, (1 - prob) / prob, random), random);
	}

	private static boolean hasExtremeMean(double size, double prob) {
		if (!Double.isFinite(size) || size <= EXTREME_MEAN_THRESHOLD
				|| !(prob < 1.0)) return false;
		double mean = size * ((1.0 - prob) / prob);
		return !Double.isFinite(mean) || mean > EXTREME_MEAN_THRESHOLD;
	}

	private static double extremeLogDensity(double x, double size, double prob) {
		double ratio = x / size;
		double onePlusRatio = 1.0 + ratio;
		double success = 1.0 / onePlusRatio;
		double failure = ratio / onePlusRatio;
		double divergence = size * onePlusRatio
				* bernoulliDivergence(success, prob);
		if (Double.isNaN(divergence)) return Double.NEGATIVE_INFINITY;
		double logTotal = log(size) + log1p(ratio);
		double logBinomial = -divergence - 0.5 * (M_LN_2PI
				+ log(size) + log(x) - logTotal);
		return -log1p(ratio) + logBinomial;
	}

	private static double bernoulliDivergence(double value, double target) {
		double complement = 1.0 - value;
		double targetComplement = 1.0 - target;
		double difference = value - target;
		if (abs(difference) <= 0.1 * min(target, targetComplement)) {
			double first = difference / target;
			double second = -difference / targetComplement;
			return target * xlog1pxMinusX(first)
					+ targetComplement * xlog1pxMinusX(second);
		}
		double result = value == 0.0 ? 0.0 : value * log(value / target);
		if (complement != 0.0)
			result += complement * log(complement / targetComplement);
		return max(0.0, result);
	}

	private static double xlog1pxMinusX(double x) {
		if (abs(x) > 1e-4) return (1.0 + x) * log1p(x) - x;
		double sum = 0.0;
		double power = x * x;
		for (int order = 2; order <= 20; order++) {
			double term = power / (order * (order - 1.0));
			sum += order % 2 == 0 ? term : -term;
			power *= x;
			if (abs(term) <= DBL_EPSILON * abs(sum)) break;
		}
		return sum;
	}

	private static double extremeCumulative(double x, double size, double prob,
			boolean lowerTail, boolean logP) {
		double observedRatio = x / size;
		double meanRatio = (1.0 - prob) / prob;
		double lower = observedRatio < meanRatio ? 0.0
				: observedRatio > meanRatio ? 1.0 : 0.5;
		double probability = lowerTail ? lower : 1.0 - lower;
		return logP ? log(probability) : probability;
	}

	public static final double random_mu(double size, double mu, RandomEngine random) {
	    if(MathFunctions.isInfinite(mu) || Double.isNaN(size) || size <= 0 || mu < 0)
	    	/* prob = 1 is ok, PR#1218 */
	    	return Double.NaN;
	    if (MathFunctions.isInfinite(size)) size = Double.MAX_VALUE / 2; // '/2' to prevent rgamma() returning Inf
	    return (mu == 0) ? 0 : Poisson.random(Gamma.random(size, mu / size, random), random);
	}

	public static final double[] random(int n, double size, double prob, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(size, prob, random);
		return rand;
	}

	public static final double[] random_mu(int n, double size, double mu, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random_mu(size, mu, random);
		return rand;
	}

	protected double size, prob;

	public NegBinomial(double size, double prob) {
		this.size = size; this.prob = prob;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, size, prob, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, size, prob, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, size, prob, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(size, prob, random);
	}

	public static final NegBinomial create_instance_from_mu(double size, double mu) {
		return new NegBinomial(size, size/(size + mu));
	}
}
