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

import jdistlib.Beta;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;
import static java.lang.Math.*;

/**
 * Extreme distribution.
 * Taken from EVD package of R
 *
 */
public class Extreme extends GenericDistribution {
	public static final double density(double x, GenericDistribution dist, int mlen, boolean largest, boolean log) {
		if (mlen <= 0)
			return Double.NaN;
		double dens = dist.density(x, true);
		if (MathFunctions.isInfinite(dens))
			return Double.NEGATIVE_INFINITY;
		double cum = dist.cumulative(x, true, log);
		if (!largest) cum = 1 - cum;
		cum = (mlen - 1) * log(cum);
		x = log(mlen) + dens + cum;
		return !log ? exp(x) : x;
	}

	public static final double cumulative(double q, GenericDistribution dist, int mlen, boolean largest, boolean lower_tail) {
		if (mlen <= 0)
			return Double.NaN;
		double distn = dist.cumulative(q, lower_tail, false);
		if (!largest) distn = 1 - distn;
		q = pow(distn, mlen);
		return largest != lower_tail ? 1 - q : q;
	}

	public static final double quantile(double p, GenericDistribution dist, int mlen, boolean largest, boolean lower_tail) {
		if (mlen <= 0)
			return Double.NaN;
		if (!lower_tail) p = 1 - p;
		return largest ? dist.quantile(pow(p, 1.0/mlen), lower_tail, false)
			: dist.quantile(1-pow(1-p, 1.0/mlen), lower_tail, false);
	}

	public static final double random(GenericDistribution dist, int mlen, boolean largest, RandomEngine random) {
		return largest ?
			dist.quantile(Beta.random(mlen, 1, random), true, false) :
			dist.quantile(Beta.random(1, mlen, random), true, false);
	}

	public static final double[] random(int n, GenericDistribution dist, int mlen, boolean largest, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(dist, mlen, largest, random);
		return rand;
	}

	protected int mlen;
	protected boolean largest;
	protected GenericDistribution dist;

	public Extreme(GenericDistribution dist, int mlen, boolean largest) {
		this.dist = dist; this.mlen = mlen; this.largest = largest;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, dist, mlen, largest, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		p = cumulative(p, dist, mlen, largest, lower_tail);
		return log_p ? log(p) : p;
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		if (log_p) q = exp(q);
		return quantile(q, dist, mlen, largest, lower_tail);
	}

	@Override
	public double random() {
		return random(dist, mlen, largest, random);
	}
}
