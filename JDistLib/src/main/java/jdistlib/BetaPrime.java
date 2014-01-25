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

/**
 * Beta Prime distribution. Also known as Beta distribution of the second kind.
 * @author Roby Joehanes
 *
 */
public class BetaPrime extends GenericDistribution {
	public static final double density(double x, double a, double b, boolean give_log) {
	    return Beta.density(x/(1-x), a, b, give_log);
	}

	public static final double cumulative(double x, double a, double b, boolean lower_tail, boolean log_p) {
		return Beta.cumulative(x/(1-x), a, b, lower_tail, log_p);
	}

	public static final double quantile(double p, double a, double b, boolean lower_tail, boolean log_p) {
		p = Beta.quantile(p, a, b, lower_tail, log_p);
		return p / (p + 1);
	}

	public static final double random(double a, double b, RandomEngine random) {
		if (a <= 0 || b <= 0) return Double.NaN;
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = Beta.quantile(u1 / 134217728, a, b, true, false);
		return u1 / (u1 + 1);
	}

	public static final double[] random(int n, double a, double b, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(a, b, random);
		return rand;
	}

	protected double a, b;

	public BetaPrime(double a, double b) {
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
