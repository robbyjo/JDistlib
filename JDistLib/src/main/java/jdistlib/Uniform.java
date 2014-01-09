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
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

public class Uniform extends GenericDistribution {
	public static final double density(double x, double a, double b, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (b <= a) return Double.NaN;

		if (a <= x && x <= b) return give_log ? -log(b - a) : 1. / (b - a);
		return (give_log ? Double.NEGATIVE_INFINITY : 0.);
	}

	public static final double cumulative(double x, double a, double b, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (b < a || MathFunctions.isInfinite(a) || MathFunctions.isInfinite(b)) return Double.NaN;
		if (x >= b) return (lower_tail ? (log_p ? 0. : 1.) : (log_p ? Double.NEGATIVE_INFINITY : 0.));
		if (x <= a) return (lower_tail ? (log_p ? Double.NEGATIVE_INFINITY : 0.) : (log_p ? 0. : 1.));
		if (lower_tail) {
			x = (x - a) / (b - a);
			return (log_p ? log(x) : (x));
		}
		x = (b - x) / (b - a);
		return (log_p ? log(x) : (x));
	}

	public static final double quantile(double p, double a, double b, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p) || Double.isNaN(a) || Double.isNaN(b)) return p + a + b;
		//R_Q_P01_check(p);
		if ((log_p	&& p > 0) || (!log_p && (p < 0 || p > 1)) ) return Double.NaN;
		if (MathFunctions.isInfinite(a) || MathFunctions.isInfinite(b) || b < a) return Double.NaN;
		if (b == a) return a;

		//return a + R_DT_qIv(p) * (b - a);
		p = (log_p ? (lower_tail ? exp(p) : - expm1(p)) : (lower_tail ? (p) : (0.5 - (p) + 0.5)));
		return a + p * (b - a);
	}

	public static final double random(double a, double b, RandomEngine random) {
		if (MathFunctions.isInfinite(a) || MathFunctions.isInfinite(b) || b < a) return Double.NaN;
		if (a == b)
			return a;
		else {
			double u;
			/* This is true of all builtin generators, but protect against user-supplied ones */
			do {u = random.nextDouble();} while (u <= 0 || u >= 1);
			return a + (b - a) * u;
		}
	}

	public static final double[] random(int n, double a, double b, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, random);
		return rand;
	}

	protected double a, b;

	public Uniform(double a, double b)
	{	this.a = a; this.b = b; }

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
