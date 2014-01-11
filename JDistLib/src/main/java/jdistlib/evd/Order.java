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
import jdistlib.math.Optimization;
import jdistlib.math.UnivariateFunction;
import jdistlib.rng.RandomEngine;
import static java.lang.Math.*;
import static jdistlib.math.MathFunctions.*;

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
		return cumulative(q, dist, mlen, j, largest, lower_tail, false);
	}

	public static final double cumulative(double q, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower_tail, boolean log_p) {
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
		double p = largest != lower_tail ? 1 - sum : sum;
		return log_p ? log(p) : p;
	}

	/**
	 * Find the quantile of order statistics. WARNING: UNTESTED!!!
	 * @param q
	 * @param dist
	 * @param mlen
	 * @param j
	 * @param largest
	 * @param lower_tail
	 * @param log_p
	 * @return Quantile
	 */
	public static final double quantile(double q, GenericDistribution dist, int mlen, int j, boolean largest, boolean lower_tail, boolean log_p) {
		if (log_p) q = exp(q);
		UnivariateFunction fun = new UnivariateFunction() {
			double q; int mlen, j; boolean largest, lower_tail, log_p;
			GenericDistribution dist;
			public void setParameters(double... params) {
				q = params[0]; mlen = (int) params[1]; j = (int) params[2];
				largest = params[3] == 0 ? false : true;
				lower_tail = params[4] == 0 ? false : true;
				log_p = params[5] == 0 ? false : true;
			}
			public void setObjects(Object... obj) {
				dist = (GenericDistribution) obj[0];
			}
			public double eval(double x) {
				double val = cumulative(x, dist, mlen, j, largest, lower_tail, log_p);
				val = val - q;
				return val * val;
			}
		};
		fun.setParameters(q, mlen, j, largest ? 1.0 : 0.0, lower_tail ? 1.0 : 0.0, log_p ? 1.0 : 0.0);
		fun.setObjects(dist);
		double min = -20, max = 20, x; // Guess within (-20, 20)
		/*
		 * The following loop assumes that the cdf of the order statistic is
		 * pretty well behaved
		 */
		while (true) {
			x = floor(Optimization.zeroin(fun, min, max, 1e-20, 10000));
			// Does the optimization returns border value?
			if (x == min) { // To the minimum side? Expand towards the negative
				max = min; min *= 2;
			} else if (x == max) { // To the maximum side? Expand towards the positive
				min = max; max *= 2;
			} else
				break;
		}
		// Numerical search is over, manual scan 
		double last_x, inc = lower_tail ? -1 : 1;
		double val = cumulative(x, dist, mlen, j, largest, lower_tail, log_p);
		while (true) {
			x += inc;
			val = cumulative(x, dist, mlen, j, largest, lower_tail, log_p);
			if (val < q) {
				last_x = x;
				x -= inc;
				break;
			}
		}
		if (last_x > x) {
			double temp = last_x; last_x = x; x = temp;
		}
		// Run another optimization round in the last interval. Hopefully fast.
		x = floor(Optimization.optimize(fun, last_x, x, 1e-20, 10000));
		return x;
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
