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

import static java.lang.Math.*;

import jdistlib.Exponential;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Fretchet distribution.
 * Taken from EVD package of R
 */
public class Fretchet extends GenericDistribution {
    public static final double density(double x, double loc, double scale, double shape, boolean log) {
        if (TailMath.invalid(loc, scale, shape) || !(shape > 0.0) || Double.isNaN(x)) return Double.NaN;
        if (x <= loc || x == Double.POSITIVE_INFINITY) return log ? Double.NEGATIVE_INFINITY : 0.0;
        double z = log(x - loc) - log(scale);
        double v = log(shape) - log(scale) - (1.0 + shape) * z - exp(-shape * z);
        return log ? v : exp(v);
    }
    public static final double cumulative(double x, double loc, double scale, double shape, boolean lower) {
        return cumulative(x, loc, scale, shape, lower, false);
    }
    public static final double cumulative(double x, double loc, double scale, double shape, boolean lower, boolean logP) {
        if (TailMath.invalid(loc, scale, shape) || !(shape > 0.0) || Double.isNaN(x)) return Double.NaN;
        double logF = x <= loc ? Double.NEGATIVE_INFINITY : -exp(-shape * (log(x - loc) - log(scale)));
        return TailMath.probability(logF, lower, logP);
    }
    public static final double quantile(double p, double loc, double scale, double shape, boolean lower) {
        return quantile(p, loc, scale, shape, lower, false);
    }
    public static final double quantile(double p, double loc, double scale, double shape, boolean lower, boolean logP) {
        if (TailMath.invalid(loc, scale, shape) || !(shape > 0.0)) return Double.NaN;
        return loc + exp(log(scale) - log(-TailMath.logLower(p, lower, logP)) / shape);
    }
    public static final double random(double loc, double scale, double shape, RandomEngine random) {
        if (TailMath.invalid(loc, scale, shape) || !(shape > 0.0)) return Double.NaN;
        return loc + exp(log(scale) - log(Exponential.random_standard(random)) / shape);
    }

	public static final double[] random(int n, double loc, double scale, double shape, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(loc, scale, shape, random);
		return rand;
	}

	protected double loc, scale, shape;

	public Fretchet(double loc, double scale, double shape) {
		this.loc = loc; this.scale = scale; this.shape = shape;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, loc, scale, shape, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, loc, scale, shape, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, loc, scale, shape, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(loc, scale, shape, random);
	}
}
