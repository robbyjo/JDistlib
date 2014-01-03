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
import jdistlib.MathFunctions;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.QRandomEngine;
import static java.lang.Math.*;
import static jdistlib.MathFunctions.*;

/**
 * Order distribution.
 * Taken from EVD package of R
 *
 */
public class Order extends GenericDistribution {
	public static final double density(double x, GenericDistribution dist, int mlen, int j, boolean largest, boolean log) {
		if (mlen <= 0 || j <= 0 || j > mlen)
			return Double.NaN;
		if (!largest)
			j = mlen + 1 - j;
		double dens = dist.density(x, true);
		if (MathFunctions.isInfinite(dens))
			return Double.NEGATIVE_INFINITY;
		double cum = dist.cumulative(x, true, log);
		cum = (mlen - j) * log(cum) + (j - 1) * log (1 - cum);
		x = lgammafn(mlen + 1) - lgammafn(j) - lgammafn(mlen - j + 1) + dens + cum;
		return !log ? exp(x) : x;
	}

	public static final double cumulative(double q, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower_tail) {
		if (mlen <= 0 || j <= 0 || j > mlen)
			return Double.NaN;
		int from = largest ? mlen + 1 - j : 0;
		double
			distn = dist.cumulative(q, lower_tail, false),
			sum = 0;
		for (int k = 1; k <= j; k++) {
			int sveck = from + k - 1;
			sum += exp(lgammafn(mlen+1) - lgammafn(sveck+1) - lgammafn(mlen - sveck + 1)
				+ sveck * log(distn) + (mlen - sveck) * log(1 - distn));
		}
		return largest != lower_tail ? 1 - sum : sum;
	}

	public static final double random(GenericDistribution dist, int mlen, int j, boolean largest, QRandomEngine random) {
		if (!largest) j = mlen + 1 - j;
		double value = Beta.random(mlen+1-j, j, random);
		return dist.quantile(value, true, false);
	}

	public static final double[] random(int n, GenericDistribution dist, int mlen, int j, boolean largest, QRandomEngine random) {
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
		p = cumulative(p, dist, mlen, j, largest, lower_tail);
		return log_p ? log(p) : p;
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		throw new RuntimeException("Not implemented, sorry!");
	}

	@Override
	public double random() {
		return random(dist, mlen, j, largest, random);
	}
}
