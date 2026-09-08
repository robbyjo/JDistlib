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

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Rayleigh distribution
 * Taken from VGAM package of R
 *
 */
public class Rayleigh extends GenericDistribution {
    public static final double density(double x, double scale, boolean log) {
        if (!(scale > 0.0) || !Double.isFinite(scale) || Double.isNaN(x)) return Double.NaN;
        if (x <= 0.0 || x == Double.POSITIVE_INFINITY) return log ? Double.NEGATIVE_INFINITY : 0.0;
        double z = x / scale;
        double v = log(x) - 2.0 * log(scale) - 0.5 * z * z;
        return log ? v : exp(v);
    }
    public static final double cumulative(double x, double scale, boolean lower) {
        return cumulative(x, scale, lower, false);
    }
    public static final double cumulative(double x, double scale, boolean lower, boolean logP) {
        if (!(scale > 0.0) || !Double.isFinite(scale) || Double.isNaN(x)) return Double.NaN;
        double z = max(x, 0.0) / scale;
        return TailMath.probability(-0.5 * z * z, !lower, logP);
    }
    public static final double quantile(double p, double scale, boolean lower) {
        return quantile(p, scale, lower, false);
    }
    public static final double quantile(double p, double scale, boolean lower, boolean logP) {
        if (!(scale > 0.0) || !Double.isFinite(scale)) return Double.NaN;
        return scale * sqrt(-2.0 * TailMath.logLower(p, !lower, logP));
    }
    public static final double random(double scale, RandomEngine random) {
        return quantile(random.nextDouble(), scale, false, false);
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
		return cumulative(p, scale, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, scale, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(scale, random);
	}
}
