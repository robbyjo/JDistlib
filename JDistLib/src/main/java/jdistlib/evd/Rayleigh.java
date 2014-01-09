/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 3 of the License.
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
package jdistlib.evd;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;
import static java.lang.Math.*;

/**
 * Rayleigh distribution
 * Taken from VGAM package of R
 *
 */
public class Rayleigh extends GenericDistribution {
	public static final double density(double x, double scale, boolean log) {
		if (scale <= 0)
			return Double.NaN;
		double v = x / scale;
		x = log(x) - 0.5 * v * v - 2 * log(scale);
		return log ? x : exp(x);
	}

	public static final double cumulative(double q, double scale, boolean lower_tail) {
		if (scale <= 0)
			return Double.NaN;
		if (q <= 0)
			return 0;
		q = q / scale;
		q = -expm1(-0.5 * q * q);
		return lower_tail ? q : 1 - q;
	}

	public static final double quantile(double p, double scale, boolean lower_tail) {
		if (scale <= 0 | p < 0 | p > 1)
			return Double.NaN;
		if (!lower_tail) p = 1 - p;
		return scale * sqrt(-2 * log1p(-p));
	}

	public static final double random(double scale, RandomEngine random) {
		return scale <= 0 ? Double.NaN : scale * sqrt(-2 * log(random.nextDouble()));
	}

	public static final double[] random(int n, double scale, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(scale, random);
		return rand;
	}

	protected double scale;

	public Rayleigh(double scale) {
		this.scale = scale;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, scale, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		p = cumulative(p, scale, lower_tail);
		return log_p ? log(p) : p;
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		if (log_p) q = exp(q);
		return quantile(q, scale, lower_tail);
	}

	@Override
	public double random() {
		return random(scale, random);
	}
}
