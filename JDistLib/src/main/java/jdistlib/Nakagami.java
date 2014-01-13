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

import static java.lang.Math.sqrt;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

public class Nakagami extends GenericDistribution {
	public static final double density(double x, double m, double omega, boolean give_log) {
	    return Gamma.density(sqrt(x), m, omega/m, give_log);
	}

	public static final double cumulative(double x, double m, double omega, boolean lower_tail, boolean log_p) {
	    return Gamma.cumulative(sqrt(x), m, omega/m, lower_tail, log_p);
	}

	public static final double quantile(double x, double m, double omega, boolean lower_tail, boolean log_p) {
	    return sqrt(Gamma.quantile(x, m, omega/m, lower_tail, log_p));
	}

	public static final double random(double m, double omega, RandomEngine random) {
	    return sqrt(Gamma.random(m, omega/m, random));
	}

	public static final double[] random(int n, double m, double omega, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(m, omega, random);
		return rand;
	}

	protected double m, omega;

	public Nakagami(double m, double omega) {
		this.m = m; this.omega = omega;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, m, omega, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, m, omega, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, m, omega, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(m, omega, random);
	}
}
