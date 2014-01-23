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
import static jdistlib.math.Constants.*;

/**
 * Bounded Arcsine distribution; bounded by [a, b]. For unbounded Arcsine distribution, use Beta(0.5, 0.5).
 * Note: Minimally tested.
 * @author Roby Joehanes
 *
 */
public class Arcsine extends GenericDistribution {
	public static final double density(double x, double a, double b, boolean give_log) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (b < a) throw new IllegalArgumentException();
		b = - M_LOG_PI - 0.5 * (log(x - a)  + log(b - x));
		return give_log ? b : exp(b);
	}

	public static final double cumulative(double x, double a, double b, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(x) || Double.isNaN(a) || Double.isNaN(b)) return x + a + b;
		if (b < a) throw new IllegalArgumentException();
		b = cumulative_raw(x, a, b);
		return log_p ? (lower_tail ? b : log1p(-exp(b))) : (lower_tail ? exp(b): 1-exp(b));
	}

	static final double cumulative_raw(double x, double a, double b) {
		return -log(M_PI_2) - log(asin(sqrt((x - a) / (b - a))));
	}

	/**
	 * Quantile method by bisection
	 * @param q
	 * @param a
	 * @param b
	 * @param lower_tail
	 * @param log_p
	 */
	public static final double quantile(double q, double a, double b, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(q) || Double.isNaN(a) || Double.isNaN(b)) return q + a + b;
		if (b < a) throw new IllegalArgumentException();
		if (log_p) q = exp(q);
		if (q < 0 || q > 1) throw new IllegalArgumentException();
		if (lower_tail) q = 1-q;
		q = log(q); // Log form comparison
		double lo = a, hi = b, f_lo = cumulative_raw(lo, a, b), f_hi = cumulative_raw(hi, a, b), mid, f_mid;
		boolean pathological = false;
		do {
			mid = (lo + hi) / 2.;
			f_mid = cumulative_raw(mid, a, b);
			// When the case is pathological, prefer to shrink the
			// upper bound when lower_tail == true (shrink the lower bound otherwise)
			if (f_mid == q && (f_hi == q || f_lo == q) && (hi - lo > 2))
				pathological = true;
			if (lower_tail) {
				if (f_lo >= q) return lo;
				if (f_mid > q) {
					hi = mid;
					f_hi = f_mid;
				} else {
					lo = mid;
					f_lo = f_mid;
				}
			} else {
				if (f_hi <= q) return hi;
				if (f_mid < q) {
					lo = mid;
					f_lo = f_mid;
				} else {
					hi = mid;
					f_hi = f_mid;
				}
			}
		} while (hi - lo > 1);
		if (pathological)
			System.err.println("Pathological case of Arcsine.quantile! Quantile estimate may not be accurate!");
		if (lower_tail)
			return f_hi <= q ? hi : f_mid <= q ? mid : lo;
		return f_lo >= q ? lo : f_mid >= q ? mid : hi;
	}

	public static final double random(double a, double b, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, a, b, true, false);
		return u1;
	}

	public static final double[] random(int n, double a, double b, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, random);
		return rand;
	}

	protected double a, b;

	public Arcsine(double a, double b) {
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
