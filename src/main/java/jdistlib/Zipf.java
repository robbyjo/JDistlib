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

import static java.lang.Double.NaN;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static jdistlib.math.MathFunctions.gharmonic;
import static jdistlib.math.MathFunctions.isInfinite;
import static jdistlib.math.MathFunctions.lgharmonic;
import static jdistlib.math.MathFunctions.logspace_add;

import jdistlib.exception.PrecisionException;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;
import jdistlib.util.Debug;

/**
 * Zipf distribution
 * Parts taken from VGAM
 *
 */
public class Zipf extends GenericDistribution {
	public static final double density(int x, int N, double s, boolean give_log) {
		if (!Double.isFinite(s)) return NaN;
		if (N <= 0 || s <= 0) return NaN;
		if (x <= 0 || x > N) return give_log ? Double.NEGATIVE_INFINITY : 0;
	    return give_log ? -s * log(x) - lgharmonic(N, s) : pow(x, -s) / gharmonic(N, s);
	}

	public static final double cumulative(int x, int N, double s, boolean lower_tail, boolean log_p) {
		if (!Double.isFinite(s)) return NaN;
		if (N <= 0 || s <= 0) return NaN;
		if (x <= 0) return DistributionUtil.boundary(false, lower_tail, log_p);
		if (x >= N) return DistributionUtil.boundary(true, lower_tail, log_p);
		if (lower_tail)
			return log_p ? lgharmonic(x, s) - lgharmonic(N, s) : gharmonic(x, s) / gharmonic(N, s);
		double sum = 0;
		if (log_p) {
			sum = Double.NEGATIVE_INFINITY;
			for (int i = x+1; i <= N; i++)
				sum = logspace_add(sum, -s * log(i));
			return sum - lgharmonic(N, s);
		}
		for (int i = x+1; i <= N; i++)
			sum += pow(i, -s);
		return sum / gharmonic(N, s);
	}

    public static final double quantile(double p, int N, double s, boolean lower, boolean logP) {
        if (N<=0 || !(s>0.0) || !Double.isFinite(s)) return NaN;
        return DistributionUtil.discreteQuantile(p,lower,logP,1,N,(x,lt,lp)->cumulative((int)x,N,s,lt,lp));
    }
	public static final double random(int N, double s, RandomEngine random) {
		if (N <= 0 || s <= 0) return Double.NaN;
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, N, s, true, false);
		return u1;
	}

	public static final double[] random(int n, int N, double s, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(N, s, random);
		return rand;
	}

	protected int N;
	protected double s;

	public Zipf(int N, double s) {
		this.N = N; this.s = s;
	}

	@Override
	public double density(double x, boolean log) {
		if (Double.isNaN(x)) return NaN;
		if (x != Math.rint(x) || x < 1 || x > N) return log ? Double.NEGATIVE_INFINITY : 0.0;
		return density((int) x, N, s, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(p)) return NaN;
		return cumulative((int) Math.floor(p), N, s, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, N, s, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(N, s, random);
	}
}
