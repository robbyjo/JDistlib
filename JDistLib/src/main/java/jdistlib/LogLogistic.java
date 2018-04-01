/*
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
import jdistlib.rng.RandomEngine;
import static java.lang.Math.*;
import static jdistlib.math.MathFunctions.log1pexp;

/**
 * Log logistic distribution. Taken from actuar package v2.3-0
 * @author Roby Joehanes
 *
 */
public class LogLogistic extends GenericDistribution {
	public static final double density(double x, double shape, double scale, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(scale)) return x + shape + scale;
		if (Double.isInfinite(shape) || Double.isInfinite(scale) || shape <= 0.0 || scale <= 0.0) return Double.NaN;
		if (x == 0.0) {
			if (shape < 1) return Double.POSITIVE_INFINITY;
			if (shape > 1) return give_log ? Double.NEGATIVE_INFINITY : 0.;
			x = 1.0 / scale;
			return (give_log ? log(x) : (x));
		}
	    double tmp, logu, log1mu;

	    tmp = shape * (log(x) - log(scale));
	    logu = - log1pexp(-tmp);
	    log1mu = - log1pexp(tmp);
	    // return ACT_D_exp(log(shape) + logu + log1mu - log(x));
	    x = log(shape) + logu + log1mu - log(x);

	    return (give_log ? (x) : exp(x));
	}

	public static final double cumulative(double q, double shape, double scale, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(q) || Double.isNaN(shape) || Double.isNaN(scale)) return q + shape + scale;
		if (Double.isInfinite(shape) || Double.isInfinite(scale) || shape <= 0.0 || scale <= 0.0) return Double.NaN;
		if (q <= 0) return lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.);
		double u = exp(-log1pexp(shape * (log(scale) - log(q))));
		//return ACT_DT_val(u);
		return (lower_tail ? (log_p  ? log(u) : (u))  : (log_p  ? log1p(-(u)) : (0.5 - (u) + 0.5)));
	}

	public static final double quantile(double p, double shape, double scale, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p) || Double.isNaN(shape) || Double.isNaN(scale)) return p + shape + scale;
		if (Double.isInfinite(shape) || Double.isInfinite(scale) || shape <= 0.0 || scale <= 0.0) return Double.NaN;

		// ACT_Q_P01_boundaries(p, 0, R_PosInf);
	    if (log_p) {
	        if(p > 0)
	            return Double.NaN;
	        if(p == 0)
	            return lower_tail ? Double.POSITIVE_INFINITY : 0;
	        if(p == Double.NEGATIVE_INFINITY)
	            return lower_tail ? 0 : Double.POSITIVE_INFINITY;
	    } else { /* !log_p */
	        if(p < 0 || p > 1)
	            return Double.NaN;
	        if(p == 0)
	            return lower_tail ? 0 : Double.POSITIVE_INFINITY;
	        if(p == 1)
	            return lower_tail ? Double.POSITIVE_INFINITY : 0;
	    }
	    // p = ACT_D_qIv(p);
	    p = (log_p  ? exp(p) : (p));
	    //return scale * R_pow(1.0 / ACT_D_Cval(p) - 1.0, 1.0/shape);

	    // p = ACT_D_Cval(p);
	    p = (lower_tail ? (log_p  ? log1p(-(p)) : (0.5 - (p) + 0.5)) : (log_p  ? log(p) : (p)));
	    return scale * pow(1.0 / p - 1.0, 1.0/shape);
	}

	public static final double random(double shape, double scale, RandomEngine random) {
		if (Double.isInfinite(shape) || Double.isInfinite(scale) || shape <= 0.0 || scale <= 0.0) return Double.NaN;
		return scale * pow(1.0 / random.nextDouble() - 1.0, 1.0 / shape);
	}

	public static final double[] random(int n, double a, double b, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, random);
		return rand;
	}

	protected double shape, scale;

	public LogLogistic(double shape, double scale) {
		this.shape = shape; this.scale = scale;
	}

	@Override
	public double density(double x, boolean log) {
		return density((int) x, shape, scale, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative((int) p, shape, scale, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, shape, scale, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(shape, scale, random);
	}
}
