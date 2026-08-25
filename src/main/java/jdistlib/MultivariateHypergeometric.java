/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.rng.RandomEngine;

/** Sampling without replacement from multiple population categories. */
public final class MultivariateHypergeometric {
	private MultivariateHypergeometric() {}

	private static double logChoose(int n, int k) {
		return lgammafn(n + 1.0) - lgammafn(k + 1.0) - lgammafn(n - k + 1.0);
	}

	public static double density(int[] x, int[] population, int draws,
			boolean giveLog) {
		if (x == null || population == null || x.length != population.length ||
				population.length < 2 || draws < 0) return Double.NaN;
		long populationSum = 0L;
		long countSum = 0L;
		double logMass = 0.0;
		for (int i = 0; i < population.length; i++) {
			if (population[i] < 0) return Double.NaN;
			populationSum += population[i];
			if (x[i] < 0 || x[i] > population[i])
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			countSum += x[i];
			logMass += logChoose(population[i], x[i]);
		}
		if (draws > populationSum || countSum != draws)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		logMass -= lgammafn(populationSum + 1.0) - lgammafn(draws + 1.0) -
				lgammafn(populationSum - draws + 1.0);
		return giveLog ? logMass : Math.exp(logMass);
	}

	public static int[] random(int[] population, int draws, RandomEngine random) {
		if (population == null || population.length < 2 || draws < 0 || random == null)
			return null;
		long totalLong = 0L;
		for (int value : population) {
			if (value < 0) return null;
			totalLong += value;
		}
		if (draws > totalLong || totalLong > Integer.MAX_VALUE) return null;
		int remainingPopulation = (int) totalLong;
		int remainingDraws = draws;
		int[] result = new int[population.length];
		for (int i = 0; i + 1 < population.length; i++) {
			result[i] = (int) HyperGeometric.random(population[i],
					remainingPopulation - population[i], remainingDraws, random);
			remainingPopulation -= population[i];
			remainingDraws -= result[i];
		}
		result[result.length - 1] = remainingDraws;
		return result;
	}

	public static int[][] random(int n, int[] population, int draws,
			RandomEngine random) {
		if (n < 0) return null;
		int[][] result = new int[n][];
		for (int i = 0; i < n; i++) result[i] = random(population, draws, random);
		return result;
	}
}
