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

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.signum;
import static jdistlib.math.Constants.M_LN2;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Laplace distribution, from VGAM package
 *
 */
public class Laplace extends GenericDistribution {
	public static final double density(double x, double location, double scale, boolean give_log) {
		if (Double.isNaN(location) || Double.isInfinite(scale)) return location + scale;
		double v = -abs(x - location) / scale - M_LN2 - log(scale);
		return give_log ? v : exp(v);
	}

	public static final double cumulative(double x, double location, double scale, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(location) || Double.isInfinite(scale)) return location + scale;
		x = (x - location) / scale;
		double p = x < 0 ? 0.5 * exp(x) : 1 - 0.5 * exp(-x);
		if (!lower_tail) p = 1 - p;
		if (log_p) p = log(p);
	    return p;
	}

	public static final double quantile(double p, double location, double scale, boolean lower_tail, boolean log_p) {
		if (log_p) p = exp(p);
		if (!lower_tail) p = 1 - p;
		return location - signum(p-0.5) * scale * (M_LN2 + log(p < 0.5 ? p : 1 - p));
	}

	public static final double random(double location, double scale, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = ((int) (134217728 * u1) + random.nextDouble()) / 134217728;
		return location - signum(u1-0.5) * scale * (M_LN2 + log(u1 < 0.5 ? u1 : 1 - u1));
	}

	public static final double[] random(int n, double location, double scale, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(location, scale, random);
		return rand;
	}

	protected double location, scale;

	public Laplace(double location, double scale) {
		this.location = location; this.scale = scale;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, location, scale, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, location, scale, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, location, scale, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(location, scale, random);
	}
}
