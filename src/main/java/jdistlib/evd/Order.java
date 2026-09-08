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
import static jdistlib.math.MathFunctions.*;

import jdistlib.Beta;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/**
 * Order distribution.
 * Taken from EVD package of R
 *
 */
public class Order extends GenericDistribution {
    public static final double density(double x, GenericDistribution dist, int mlen, int j, boolean largest, boolean log) {
        if (mlen <= 0 || j <= 0 || j > mlen) return Double.NaN;
        int k = largest ? mlen + 1 - j : j;
        if (mlen == 1) return dist.density(x, log);
        double value = lgammafn(mlen + 1.0) - lgammafn(k) - lgammafn(mlen - k + 1.0) + dist.density(x, true);
        if (k > 1) value += (k - 1.0) * dist.cumulative(x, true, true);
        if (k < mlen) value += (mlen - k) * dist.cumulative(x, false, true);
        return log ? value : exp(value);
    }
    public static final double cumulative(double x, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower) {
        return cumulative(x, dist, mlen, j, largest, lower, false);
    }
    public static final double cumulative(double x, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower, boolean logP) {
        if (mlen <= 0 || j <= 0 || j > mlen) return Double.NaN;
        int k = largest ? mlen + 1 - j : j;
        double lf = dist.cumulative(x, true, true), ls = dist.cumulative(x, false, true);
        if (lf <= ls) return Beta.cumulative(exp(lf), k, mlen - k + 1.0, lower, logP);
        return Beta.cumulative(exp(ls), mlen - k + 1.0, k, !lower, logP);
    }
    /** Quantile from the beta distribution of the transformed order statistic. */
    public static final double quantile(double p, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower, boolean logP) {
        if (mlen <= 0 || j <= 0 || j > mlen) return Double.NaN;
        int k = largest ? mlen + 1 - j : j;
        double v = Beta.quantile(p, k, mlen - k + 1.0, lower, logP);
        if (v <= 0.5) return dist.quantile(v, true, false);
        v = Beta.quantile(p, mlen - k + 1.0, k, !lower, logP);
        return dist.quantile(v, false, false);
    }
	public static final double random(GenericDistribution dist, int mlen, int j, boolean largest, RandomEngine random) {
		if (!largest) j = mlen + 1 - j;
		double value = Beta.random(mlen+1-j, j, random);
		return dist.quantile(value, true, false);
	}

	public static final double[] random(int n, GenericDistribution dist, int mlen, int j, boolean largest, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(dist, mlen, j, largest, random);
		return rand;
	}

	protected int mlen, j;
	protected boolean largest;
	protected GenericDistribution dist;

	public Order(GenericDistribution dist, int mlen, int j, boolean largest) {
		this.dist = dist; this.mlen = mlen; this.j = j; this.largest = largest;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, dist, mlen, j, largest, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, dist, mlen, j, largest, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, dist, mlen, j, largest, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(dist, mlen, j, largest, random);
	}
}
