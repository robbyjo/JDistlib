/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Multivariate Cauchy distribution, the multivariate Student t law with one df. */
public final class MultivariateCauchy {
	private MultivariateCauchy() {}

	public static double density(double[] x, double[] location, double[][] scale,
			boolean giveLog) {
		return MultivariateStudentT.density(x, location, scale, 1.0, giveLog);
	}

	public static double[] random(double[] location, double[][] scale,
			RandomEngine random) {
		return MultivariateStudentT.random(location, scale, 1.0, random);
	}

	public static double[][] random(int n, double[] location, double[][] scale,
			RandomEngine random) {
		return MultivariateStudentT.random(n, location, scale, 1.0, random);
	}
}
