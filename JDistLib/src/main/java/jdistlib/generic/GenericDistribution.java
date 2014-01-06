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
package jdistlib.generic;

import jdistlib.rng.QMersenneTwister;
import jdistlib.rng.QRandomEngine;

/**
 * An interface for a generic distribution. All parameters have to be encoded (either as fields or otherwise).
 * Treat this interface as an adapter to the other distributions.
 * 
 * @author Roby Joehanes
 *
 */
public abstract class GenericDistribution {
	protected QRandomEngine random = new QMersenneTwister();
	public abstract double density(double x, boolean log);
	public abstract double cumulative(double p, boolean lower_tail, boolean log_p);
	public abstract double quantile(double q, boolean lower_tail, boolean log_p);
	public abstract double random();

	public double[] random(int n) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random();
		return rand;
	}

	public void setRandomEngine(QRandomEngine r) {
		random = r;
	}

	public QRandomEngine getRandomEngine() {
		return random;
	}

	/**
	 * Old RNG API
	 * @deprecated
	 * @param r random number generator
	 * @return Random number for the distribution
	 */
	public double random(QRandomEngine r) {
		QRandomEngine temp = random;
		random = r;
		double v = random();
		random = temp;
		return v;
	}
}
