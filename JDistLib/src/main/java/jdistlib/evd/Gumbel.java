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

import static java.lang.Math.exp;
import static java.lang.Math.log;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Gumbel distribution. It is a special case of Extreme Value Distribution with shape == 0.
 * Taken from EVD package of R
 *
 */
public class Gumbel extends GenericDistribution {
	public static final double density(double x, double loc, double scale, boolean log)
	{	return GEV.density(x, loc, scale, 0, log); }

	public static final double cumulative(double q, double loc, double scale, boolean lower_tail)
	{	return GEV.cumulative(q, loc, scale, 0, lower_tail); }

	public static final double quantile(double p, double loc, double scale, boolean lower_tail)
	{	return GEV.quantile(p, loc, scale, 0, lower_tail); }

	public static final double random(double loc, double scale, RandomEngine random)
	{	return GEV.random(loc, scale, 0, random); }

	public static final double[] random(int n, double loc, double scale, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(loc, scale, random);
		return rand;
	}

	protected double loc, scale;

	public Gumbel(double loc, double scale) {
		this.loc = loc; this.scale = scale;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, loc, scale, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		p = cumulative(p, loc, scale, lower_tail);
		return log_p ? log(p) : p;
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		if (log_p) q = exp(q);
		return quantile(q, loc, scale, lower_tail);
	}

	@Override
	public double random() {
		return random(loc, scale, random);
	}
}
