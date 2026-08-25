/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Component-wise exponential transform of a multivariate normal vector. */
public final class MultivariateLogNormal {
	private MultivariateLogNormal() {}

	public static double density(double[] x, double[] meanLog,
			double[][] covarianceLog, boolean giveLog) {
		if (x == null) return Double.NaN;
		double[] logged = new double[x.length];
		double jacobian = 0.0;
		for (int i = 0; i < x.length; i++) {
			if (!(x[i] > 0.0) || !Double.isFinite(x[i]))
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			logged[i] = Math.log(x[i]);
			jacobian += logged[i];
		}
		double logDensity = MultivariateNormal.density(logged, meanLog,
				covarianceLog, true) - jacobian;
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] meanLog, double[][] covarianceLog,
			RandomEngine random) {
		double[] result = MultivariateNormal.random(meanLog, covarianceLog, random);
		if (result == null) return null;
		for (int i = 0; i < result.length; i++) result[i] = Math.exp(result[i]);
		return result;
	}

	public static double[][] random(int n, double[] meanLog,
			double[][] covarianceLog, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(meanLog, covarianceLog, random);
		return result;
	}
}
