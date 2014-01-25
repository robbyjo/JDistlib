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

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static jdistlib.math.MathFunctions.logspace_sub;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Kumaraswamy distribution
 * @author Roby Joehanes
 *
 */
public class Kumaraswamy extends GenericDistribution {
	public static final double density(double x, double a, double b, boolean give_log) {
		if (a <= 0 || b <= 0) return Double.NaN;
		if (x < 0 || x > 1) return 0;
		x = log(a) + log(b) + (a - 1) * log(x) + (b - 1) * log1p(-pow(x, a));
	    return give_log ? x : exp(x);
	}

	public static final double cumulative(double x, double a, double b, boolean lower_tail, boolean log_p) {
		if (a <= 0 || b <= 0) return Double.NaN;
		if (x < 0) return 0;
		if (x > 1) return 1;
		x = b * log1p(-pow(x, a));
		return log_p ? (lower_tail ? logspace_sub(0, x) : x) : (lower_tail ? 1 - exp(x) : x);
	}

	public static final double quantile(double p, double a, double b, boolean lower_tail, boolean log_p) {
		if (a <= 0 || b <= 0) return Double.NaN;
		if (log_p) {
			if (p > 0) return Double.NaN;
			if (p == 0) return lower_tail ? 1 : 0;
			if (p == Double.NEGATIVE_INFINITY) return lower_tail ? 0 : 1;
		} else {
			if (p < 0 || p > 1) return Double.NaN;
			if (p == 0) return lower_tail ? 0 : 1;
			if (p == 1) return lower_tail ? 1 : 0;
		}
		if (log_p) p = exp(p);
		return !lower_tail ? pow(1-pow(1-p, 1.0/b), 1.0/a) : pow(1-pow(p, 1.0/b), 1.0/a);
	}

	public static final double random(double a, double b, RandomEngine random) {
		if (a <= 0 || b <= 0) return Double.NaN;
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = pow(1-pow(u1 / 134217728, 1.0/b), 1.0/a);
		return u1;
	}

	public static final double[] random(int n, double a, double b, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, random);
		return rand;
	}

	protected double a, b;

	public Kumaraswamy(double a, double b) {
		this.a = a; this.b = b;
	}

	@Override
	public double density(double x, boolean log) {
		return density((int) x, a, b, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative((int) p, a, b, lower_tail, log_p);
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
