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

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static jdistlib.math.Constants.M_LN2;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Skewed T distribution, from skewt package
 *
 */
public class SkewedT extends GenericDistribution {
	public static final double density(double x, double df, double gamma, boolean give_log) {
		if (Double.isNaN(df) || Double.isInfinite(gamma)) return df + gamma;
		double
			v = gamma + 1./gamma,
			dt = T.density(x < 0 ? (gamma * x) : (x / gamma), df, give_log);
		return give_log ? M_LN2 - log(v) + dt : (2 / v) * dt;
	}

	public static final double cumulative(double x, double df, double gamma, boolean lower_tail, boolean log_p) {
		if (Double.isNaN(df) || Double.isInfinite(gamma)) return df + gamma;
		double v = gamma * gamma, pt;
		if (x < 0) {
			pt = 2 / (v + 1) * T.cumulative(gamma * x, df, true, false);
		} else {
			pt = 1/(v+1) + 2 / (1 + (1/v)) * (T.cumulative(x / gamma, df, true, false) - 0.5);
		}
		if (!lower_tail) pt = 1 - pt;
		if (log_p) pt = log(pt);
	    return pt;
	}

	public static final double quantile(double p, double df, double gamma, boolean lower_tail, boolean log_p) {
		if (log_p) p = exp(p);
		if (!lower_tail) p = 1 - p;
		double v = gamma * gamma;
		double p0 = cumulative(0, df, gamma, true, false);
	    return p < p0 ? (1/gamma) * T.quantile((v+1) * p/2, df, true, false)
	    	: gamma * T.quantile((1 + 1/v) / 2 * (p - p0) + 0.5, df, true, false);
	}

	public static final double random(double df, double gamma, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, df, gamma, true, false);
		return u1;
	}

	public static final double[] random(int n, double df, double gamma, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(df, gamma, random);
		return rand;
	}

	protected double df, gamma;

	public SkewedT(double df, double gamma) {
		this.df = df; this.gamma = gamma;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, df, gamma, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, df, gamma, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, df, gamma, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(df, gamma, random);
	}
}
