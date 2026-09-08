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

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

public class Weibull extends GenericDistribution {
	public static final double density(double x, double shape, double scale, boolean give_log) {
		double tmp1, tmp2;
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(scale)) return x + shape + scale;
		if (shape <= 0 || scale <= 0) return Double.NaN;

		if (x < 0) return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		if (MathFunctions.isInfinite(x)) return (give_log ? Double.NEGATIVE_INFINITY : 0.);
		if (x == 0) {
			if (shape < 1) return Double.POSITIVE_INFINITY;
			if (shape > 1) return give_log ? Double.NEGATIVE_INFINITY : 0.;
			return give_log ? -log(scale) : 1 / scale;
		}
		if (Double.isInfinite(scale) && Double.isFinite(shape))
			return give_log ? Double.NEGATIVE_INFINITY : 0.;
		if (Double.isInfinite(shape) && x == scale)
			return Double.POSITIVE_INFINITY;
		double ratio = x / scale;
		tmp1 = pow(ratio, shape - 1);
		tmp2 = tmp1 * ratio;
		double factor = shape * tmp1 / scale;
		double result = give_log ? -tmp2 + log(factor) : factor * exp(-tmp2);
		if (ratio >= Double.MIN_NORMAL && Double.isFinite(ratio)
				&& Double.isFinite(tmp1) && tmp1 >= Double.MIN_NORMAL && Double.isFinite(factor) && factor >= Double.MIN_NORMAL
				&& Double.isFinite(result) && (give_log || result > 0)) return result;
		double logRatio = ratio >= Double.MIN_NORMAL && Double.isFinite(ratio)
				? log(ratio) : log(x) - log(scale);
		double power = exp(shape * logRatio);
		if (Double.isInfinite(power)) return give_log ? Double.NEGATIVE_INFINITY : 0.;
		double logDensity = log(shape) - log(scale) + (shape - 1) * logRatio - power;
		return give_log ? logDensity : exp(logDensity);
	}

	public static final double cumulative(double x, double shape, double scale, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(x) || Double.isNaN(shape) || Double.isNaN(scale)) return x + shape + scale;
		if (shape <= 0 || scale <= 0) return Double.NaN;

		if (x <= 0)	return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		double ratio = x / scale;
		double power = pow(ratio, shape);
		if (Double.isFinite(x) && Double.isFinite(shape) && Double.isFinite(scale)
				&& (ratio < Double.MIN_NORMAL || Double.isInfinite(ratio) || power < Double.MIN_NORMAL)) {
			double logPower = shape * (log(x) - log(scale));
			if (lower_tail && log_p && logPower < log(Double.MIN_NORMAL)) return logPower;
			power = exp(logPower);
		}
		x = -power;
		if (lower_tail)
			return (log_p
					/* log(1 - exp(x))  for x < 0 : */
					//? R_Log1_Exp(x) : -expm1(x));
					? ((x) > -M_LN2 ? log(-expm1(x)) : log1p(-exp(x))) : -expm1(x));
		/* else:  !lower_tail */
		//return R_D_exp(x);
		return (log_p ? (x) : exp(x));
	}

	public static final double quantile(double p, double shape, double scale, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p) || Double.isNaN(shape) || Double.isNaN(scale)) return p + shape + scale;
		if (shape <= 0 || scale <= 0) return Double.NaN;

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

		//return scale * pow(- R_DT_Clog(p), 1./shape) ;
		double hazard = -(lower_tail? (log_p ? ((p) > -M_LN2 ? log(-expm1(p)) : log1p(-exp(p))) : log1p(-p)) : (log_p ? (p) : log(p)));
		if (log_p && lower_tail && hazard < Double.MIN_NORMAL && Double.isFinite(scale)) {
			double result = scale * exp(p / shape);
			return result == 0 || Double.isInfinite(result) ? exp(log(scale) + p / shape) : result;
		}
		double result = scale * pow(hazard, 1. / shape);
		if ((result == 0 || Double.isInfinite(result)) && Double.isFinite(scale)) {
			double logHazard = log_p && lower_tail && hazard == 0 ? p : log(hazard);
			return exp(log(scale) + logHazard / shape);
		}
		return result;
	}

	public static final double random(double shape, double scale, RandomEngine random) {
		if (MathFunctions.isInfinite(shape) || MathFunctions.isInfinite(scale) || shape <= 0. || scale <= 0.) {
			if(scale == 0.) return 0.;
			/* else */
			return Double.NaN;
		}
		return scale * pow(-log(random.nextDouble()), 1.0 / shape);
	}

	public static final double[] random(int n, double shape, double scale, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(shape, scale, random);
		return rand;
	}

	protected double shape, scale;

	public Weibull(double shape, double scale) {
		this.shape = shape; this.scale = scale;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, shape, scale, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, shape, scale, lower_tail, log_p);
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
