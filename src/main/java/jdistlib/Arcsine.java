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

import static java.lang.Math.*;
import static jdistlib.math.Constants.*;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;
import jdistlib.util.Debug;

/**
 * Bounded Arcsine distribution; bounded by [a, b]. For unbounded Arcsine distribution, use Beta(0.5, 0.5).
 * Note: Minimally tested.
 * @author Roby Joehanes
 *
 */
public class Arcsine extends GenericDistribution {
    private static boolean invalid(double a, double b) {
        return !Double.isFinite(a) || !Double.isFinite(b) || !(a < b);
    }
    public static final double density(double x, double a, double b, boolean giveLog) {
        if (invalid(a, b) || Double.isNaN(x)) return Double.NaN;
        if (x < a || x > b) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
        double value = -log(PI) - 0.5 * (log(x - a) + log(b - x));
        return giveLog ? value : exp(value);
    }
    public static final double cumulative(double x, double a, double b, boolean lower, boolean logP) {
        if (invalid(a, b) || Double.isNaN(x)) return Double.NaN;
        if (x <= a) return DistributionUtil.boundary(false, lower, logP);
        if (x >= b) return DistributionUtil.boundary(true, lower, logP);
        // The two arcsine integrals evaluate opposite tails directly.
        double ratio = lower ? (x - a) / (b - a) : (b - x) / (b - a);
        double value = (2.0 / PI) * asin(sqrt(ratio));
        return logP ? log(value) : value;
    }
    static final double cumulative_raw(double x, double a, double b) {
        return cumulative(x, a, b, true, true);
    }
    /** Exact inverse of the arcsine CDF. */
    public static final double quantile(double p, double a, double b, boolean lower, boolean logP) {
        if (invalid(a, b) || Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
        double lp = logP ? p : log(p);
        boolean direct = lower;
        if (lp > -log(2.0)) { lp = DistributionUtil.logOneMinusExp(lp); direct = !direct; }
        double sine = sin(PI * 0.5 * exp(lp));
        double fraction = sine * sine;
        return direct ? a + (b - a) * fraction : b - (b - a) * fraction;
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
