/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Dirichlet-multinomial (multivariate Pólya) distribution. */
public final class DirichletMultinomial {
	private DirichletMultinomial() {}

	public static double density(double[] x, int size, double[] alpha,
			boolean giveLog) {
		if (x == null || alpha == null || x.length != alpha.length ||
				alpha.length < 2 || size < 0) return Double.NaN;
		double alphaSum = 0.0;
		double countSum = 0.0;
		double logMass = lgammafn(size + 1.0);
		for (int i = 0; i < alpha.length; i++) {
			if (!(alpha[i] > 0.0) || !Double.isFinite(alpha[i])) return Double.NaN;
			if (!(x[i] >= 0.0) || x[i] != Math.rint(x[i]) ||
					!Double.isFinite(x[i]))
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			alphaSum += alpha[i];
			countSum += x[i];
			logMass -= lgammafn(x[i] + 1.0);
			logMass += lgammafn(x[i] + alpha[i]) - lgammafn(alpha[i]);
		}
		if (countSum != size) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		logMass += lgammafn(alphaSum) - lgammafn(size + alphaSum);
		return giveLog ? logMass : Math.exp(logMass);
	}

	public static int[] random(int size, double[] alpha, RandomEngine random) {
		if (size < 0) return null;
		double[] probabilities = Dirichlet.random(alpha, random);
		return probabilities == null ? null : Multinomial.random(size, probabilities, random);
	}

	public static int[][] random(int n, int size, double[] alpha,
			RandomEngine random) {
		if (n < 0) return null;
		int[][] result = new int[n][];
		for (int i = 0; i < n; i++) result[i] = random(size, alpha, random);
		return result;
	}
}
