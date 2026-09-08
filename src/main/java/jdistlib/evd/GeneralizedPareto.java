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
 * Generalized Pareto Distribution
 * Taken from EVD package of R
 *
 */
public class GeneralizedPareto extends GenericDistribution {
    public static final double density(double x, double loc, double scale, double shape, boolean log) {
        if (TailMath.invalid(loc, scale, shape) || Double.isNaN(x)) return Double.NaN;
        double z = (x - loc) / scale;
        if (z < 0.0 || z == Double.POSITIVE_INFINITY || shape * z < -1.0)
            return log ? Double.NEGATIVE_INFINITY : 0.0;
        double v;
        if (shape * z == -1.0) {
            if (shape == -1.0) return log ? -log(scale) : 1.0 / scale;
            return shape < -1.0 ? Double.POSITIVE_INFINITY : (log ? Double.NEGATIVE_INFINITY : 0.0);
        }
        v = -log(scale) - (shape == 0.0 ? z : (1.0 + shape) * log1p(shape * z) / shape);
        return log ? v : exp(v);
    }
    public static final double cumulative(double x, double loc, double scale, double shape, boolean lower) {
        return cumulative(x, loc, scale, shape, lower, false);
    }
    public static final double cumulative(double x, double loc, double scale, double shape, boolean lower, boolean logP) {
        if (TailMath.invalid(loc, scale, shape) || Double.isNaN(x)) return Double.NaN;
        double z = max((x - loc) / scale, 0.0);
        double logS = shape == 0.0 ? -z : (shape * z <= -1.0 ? Double.NEGATIVE_INFINITY : -log1p(shape * z) / shape);
        return TailMath.probability(logS, !lower, logP);
    }
    public static final double quantile(double p, double loc, double scale, double shape, boolean lower) {
        return quantile(p, loc, scale, shape, lower, false);
    }
    public static final double quantile(double p, double loc, double scale, double shape, boolean lower, boolean logP) {
        if (TailMath.invalid(loc, scale, shape)) return Double.NaN;
        double logS = TailMath.logLower(p, !lower, logP);
        return loc + scale * (shape == 0.0 ? -logS : expm1(-shape * logS) / shape);
    }
    public static final double random(double loc, double scale, double shape, RandomEngine random) {
        if (TailMath.invalid(loc, scale, shape)) return Double.NaN;
        double e = Exponential.random_standard(random);
        return loc + scale * (shape == 0.0 ? e : expm1(shape * e) / shape);
    }

	public static final double[] random(int n, double loc, double scale, double shape, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(loc, scale, shape, random);
		return rand;
	}

	protected double loc, scale, shape;

	public GeneralizedPareto(double loc, double scale, double shape) {
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
