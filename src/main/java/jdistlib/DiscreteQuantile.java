/*
 * Mathlib-derived discrete quantile search, based on R 4.6.1
 * src/nmath/qDiscrete_search.h. GPL-2.0-or-later.
 */
package jdistlib;

import static java.lang.Math.floor;

final class DiscreteQuantile {
	interface Cumulative {
		double value(double y, boolean lowerTail, boolean logProbability);
	}

	private DiscreteQuantile() {}

	static double quantile(double p, boolean lowerTail, boolean logProbability,
			double mean, double sigma, double skewness, double maximum,
			Cumulative cumulative) {
		double z = Normal.quantile(p, 0, 1, lowerTail, logProbability);
		double y = Math.rint(mean + sigma * (z + skewness * (z * z - 1) / 6));
		if (y < 0)
			y = 0;
		if (y > maximum)
			y = maximum;

		z = cumulative.value(y, lowerTail, logProbability);
		if (logProbability) {
			double fuzz = 2 * Math.ulp(1.0);
			if (lowerTail && p > -Double.MAX_VALUE)
				p *= 1 + fuzz;
			else
				p *= 1 - fuzz;
		} else {
			double fuzz = 8 * Math.ulp(1.0);
			if (lowerTail)
				p *= 1 - fuzz;
			else if (1 - p > 4 * fuzz)
				p *= 1 + fuzz;
		}

		if (y < 4096)
			return search(y, z, p, 1, maximum, lowerTail, logProbability, cumulative);

		double increment = floor(y / 64);
		double oldIncrement;
		do {
			oldIncrement = increment;
			y = search(y, z, p, increment, maximum, lowerTail, logProbability, cumulative);
			z = cumulative.value(y, lowerTail, logProbability);
			increment = Math.max(1, floor(increment / 8));
		} while (oldIncrement > 1 && increment > y * 1e-15);
		return y;
	}

	private static double search(double y, double z, double p, double increment,
			double maximum, boolean lowerTail, boolean logProbability,
			Cumulative cumulative) {
		boolean left = lowerTail ? z >= p : z < p;
		if (left) {
			for (;;) {
				double newz = Double.NaN;
				if (y > 0)
					newz = cumulative.value(y - increment, lowerTail, logProbability);
				else if (y < 0)
					y = 0;
				if (y == 0 || Double.isNaN(newz) || (lowerTail ? newz < p : newz >= p))
					return y;
				y = Math.max(0, y - increment);
				z = newz;
			}
		}

		for (;;) {
			double previous = y;
			double next = y + increment;
			if (next <= y)
				return y;
			y = Math.min(next, maximum);
			double newz = cumulative.value(y, lowerTail, logProbability);
			if (y == maximum || Double.isNaN(newz) || (lowerTail ? newz >= p : newz < p))
				return increment <= 1 ? y : previous;
			z = newz;
		}
	}
}
