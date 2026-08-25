/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Sum of independent Bernoulli trials with unequal success probabilities. */
public class PoissonBinomial extends GenericDistribution {
	private static double[] mass(double[] probabilities) {
		if (probabilities == null) return null;
		double[] result = new double[probabilities.length + 1];
		result[0] = 1.0;
		int used = 0;
		for (double probability : probabilities) {
			if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
				return null;
			}
			used++;
			for (int k = used; k >= 1; k--) {
				result[k] = result[k] * (1.0 - probability)
						+ result[k - 1] * probability;
			}
			result[0] *= 1.0 - probability;
		}
		return result;
	}

	public static double density(double x, double[] probabilities,
			boolean giveLog) {
		double[] pmf = mass(probabilities);
		if (Double.isNaN(x) || pmf == null) return Double.NaN;
		if (x < 0.0 || x >= pmf.length || x != Math.rint(x)) {
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = pmf[(int) x];
		return giveLog ? Math.log(result) : result;
	}

	public static double cumulative(double x, double[] probabilities,
			boolean lowerTail, boolean logP) {
		double[] pmf = mass(probabilities);
		if (Double.isNaN(x) || pmf == null) return Double.NaN;
		double result = 0.0;
		int last = Math.min((int) Math.floor(x), pmf.length - 1);
		if (lowerTail) {
			if (x >= 0.0) for (int i = 0; i <= last; i++) result += pmf[i];
		} else {
			for (int i = Math.max(0, last + 1); i < pmf.length; i++) result += pmf[i];
		}
		return logP ? Math.log(result) : result;
	}

	public static double quantile(double p, double[] probabilities,
			boolean lowerTail, boolean logP) {
		if (probabilities == null) return Double.NaN;
		return DistributionUtil.discreteQuantile(p, lowerTail, logP, 0.0,
				probabilities.length,
				(x, lt, lp) -> cumulative(x, probabilities, lt, lp));
	}

	public static double random(double[] probabilities, RandomEngine random) {
		if (mass(probabilities) == null) return Double.NaN;
		int result = 0;
		for (double probability : probabilities) {
			if (random.nextDouble() < probability) result++;
		}
		return result;
	}

	public static double[] random(int n, double[] probabilities,
			RandomEngine random) {
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = random(probabilities, random);
		return result;
	}

	private final double[] probabilities;
	private final double[] pmf;

	public PoissonBinomial(double[] probabilities) {
		this.probabilities = probabilities.clone();
		this.pmf = mass(this.probabilities);
	}

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x) || pmf == null) return Double.NaN;
		if (x < 0.0 || x >= pmf.length || x != Math.rint(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double result = pmf[(int) x];
		return log ? Math.log(result) : result;
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || pmf == null) return Double.NaN;
		double result = 0.0;
		int last = Math.min((int) Math.floor(x), pmf.length - 1);
		if (lowerTail) {
			if (x >= 0.0) for (int i = 0; i <= last; i++) result += pmf[i];
		} else {
			for (int i = Math.max(0, last + 1); i < pmf.length; i++) result += pmf[i];
		}
		return logP ? Math.log(result) : result;
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return DistributionUtil.discreteQuantile(p, lowerTail, logP, 0.0,
				probabilities.length, this::cumulative);
	}
	@Override public double random() { return random(probabilities, random); }
}
