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

/** Reciprocal-gamma law: if Y has Gamma(shape, scale), X = 1/Y.
 * The conventional inverse-gamma scale is therefore 1/scale. */
public class InvGamma extends GenericDistribution {
	public static final double density(double x, double shape, double scale, boolean give_log) {
		if (!(shape > 0.0) || !(scale > 0.0) || !Double.isFinite(shape)
				|| !Double.isFinite(scale) || Double.isNaN(x)) return Double.NaN;
		if (x <= 0.0 || x == Double.POSITIVE_INFINITY) return give_log ? Double.NEGATIVE_INFINITY : 0.0;
		double value = Gamma.density(1.0 / x, shape, scale, true) - 2.0 * Math.log(x);
		return give_log ? value : Math.exp(value);
	}

	public static final double cumulative(double x, double alph, double scale, boolean lower_tail, boolean log_p) {
		if (!(alph > 0.0) || !(scale > 0.0) || !Double.isFinite(alph)
				|| !Double.isFinite(scale) || Double.isNaN(x)) return Double.NaN;
		if (x <= 0.0) return DistributionUtil.boundary(false, lower_tail, log_p);
	    return Gamma.cumulative(1/x, alph, scale, !lower_tail, log_p);
	}

	public static final double quantile(double x, double alph, double scale, boolean lower_tail, boolean log_p) {
	    return 1./Gamma.quantile(x, alph, scale, !lower_tail, log_p);
	}

	public static final double random(double alph, double scale, RandomEngine random) {
	    return 1./Gamma.random(alph, scale, random);
	}

	public static final double[] random(int n, double alph, double scale, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(alph, scale, random);
		return rand;
	}

	protected double shape, scale;

	public InvGamma(double shape, double scale) {
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
