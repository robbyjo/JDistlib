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

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] scale,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return MultivariateStudentT.probability(lower, upper, location, scale, 1.0,
				options, random);
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] location, double[][] scale) {
		return MultivariateStudentT.probability(lower, upper, location, scale, 1.0);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scale,
			MultivariateProbabilityOptions options, RandomEngine random) {
		return MultivariateStudentT.cumulative(upper, location, scale, 1.0,
				options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] location, double[][] scale) {
		return MultivariateStudentT.cumulative(upper, location, scale, 1.0);
	}

	public static double equicoordinateQuantile(double p, double[] location,
			double[][] scale, MultivariateProbabilityOptions options,
			RandomEngine random) {
		return MultivariateStudentT.equicoordinateQuantile(p, location, scale, 1.0,
				options, random);
	}

	public static double equicoordinateQuantile(double p, double[] location,
			double[][] scale) {
		return MultivariateStudentT.equicoordinateQuantile(p, location, scale, 1.0);
	}

	public static double radialQuantile(double p, int dimension,
			boolean lowerTail, boolean logProbability) {
		return MultivariateStudentT.radialQuantile(p, dimension, 1.0, lowerTail,
				logProbability);
	}
}
