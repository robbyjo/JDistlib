/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static jdistlib.math.MathFunctions.lgammafn;

import java.util.Arrays;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Dirichlet distribution on a probability simplex. */
public final class Dirichlet {
	private static final long PROBABILITY_SEED = 0x4469726963686c74L;
	private Dirichlet() {}

	private static boolean validAlpha(double[] alpha) {
		if (alpha == null || alpha.length < 2) return false;
		for (double value : alpha)
			if (!(value > 0.0) || !Double.isFinite(value)) return false;
		return true;
	}

	public static double density(double[] x, double[] alpha, boolean giveLog) {
		if (!validAlpha(alpha) || x == null || x.length != alpha.length)
			return Double.NaN;
		double alphaSum = 0.0;
		double xSum = 0.0;
		double logDensity = 0.0;
		boolean zeroWithSmallShape = false;
		boolean zeroWithLargeShape = false;
		for (int i = 0; i < x.length; i++) {
			if (!(x[i] >= 0.0) || !Double.isFinite(x[i]))
				return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
			xSum += x[i];
			alphaSum += alpha[i];
			logDensity -= lgammafn(alpha[i]);
			if (x[i] == 0.0) {
				if (alpha[i] < 1.0) zeroWithSmallShape = true;
				else if (alpha[i] > 1.0) zeroWithLargeShape = true;
			} else {
				logDensity += (alpha[i] - 1.0) * Math.log(x[i]);
			}
		}
		double tolerance = 16.0 * Math.ulp(1.0) * x.length;
		if (Math.abs(xSum - 1.0) > tolerance)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		if (zeroWithSmallShape && zeroWithLargeShape) return Double.NaN;
		if (zeroWithSmallShape)
			return Double.POSITIVE_INFINITY;
		if (zeroWithLargeShape)
			return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
		logDensity += lgammafn(alphaSum);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	public static double[] random(double[] alpha, RandomEngine random) {
		if (!validAlpha(alpha) || random == null) return null;
		double[] result = new double[alpha.length];
		for (int attempt = 0; attempt < 100; attempt++) {
			double sum = 0.0;
			for (int i = 0; i < alpha.length; i++) {
				result[i] = Gamma.random(alpha[i], 1.0, random);
				sum += result[i];
			}
			if (sum > 0.0 && Double.isFinite(sum)) {
				for (int i = 0; i < result.length; i++) result[i] /= sum;
				return result;
			}
		}
		return null;
	}

	public static double[][] random(int n, double[] alpha, RandomEngine random) {
		if (n < 0) return null;
		double[][] result = new double[n][];
		for (int i = 0; i < n; i++) result[i] = random(alpha, random);
		return result;
	}

	/**
	 * Computes a simplex-aware rectangle probability. The returned error is the
	 * randomized-replication integration indicator used by the other
	 * multivariate probability APIs.
	 */
	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] alpha, MultivariateProbabilityOptions options,
			RandomEngine random) {
		if (!validAlpha(alpha) || lower == null || upper == null ||
				lower.length != alpha.length || upper.length != alpha.length)
			return invalidProbability();
		final int dimension = alpha.length;
		final double[] lo = new double[dimension];
		final double[] hi = new double[dimension];
		double lowerSum = 0.0;
		double upperSum = 0.0;
		boolean unrestricted = true;
		for (int i = 0; i < dimension; i++) {
			if (Double.isNaN(lower[i]) || Double.isNaN(upper[i]))
				return invalidProbability();
			lo[i] = Math.max(0.0, lower[i]);
			hi[i] = Math.min(1.0, upper[i]);
			if (!(lo[i] < hi[i])) return exactProbability(0.0);
			lowerSum += lo[i];
			upperSum += hi[i];
			unrestricted &= lower[i] <= 0.0 && upper[i] >= 1.0;
		}
		if (unrestricted) return exactProbability(1.0);
		if (!(lowerSum < 1.0) || !(upperSum > 1.0))
			return exactProbability(0.0);

		final double[] remainingAlpha = new double[dimension];
		final double[] futureLower = new double[dimension];
		final double[] futureUpper = new double[dimension];
		for (int i = dimension - 1; i >= 0; i--) {
			remainingAlpha[i] = alpha[i] + (i + 1 < dimension ?
					remainingAlpha[i + 1] : 0.0);
			futureLower[i] = lo[i] + (i + 1 < dimension ? futureLower[i + 1] : 0.0);
			futureUpper[i] = hi[i] + (i + 1 < dimension ? futureUpper[i + 1] : 0.0);
		}
		if (dimension == 2) {
			double left = Math.max(lo[0], 1.0 - hi[1]);
			double right = Math.min(hi[0], 1.0 - lo[1]);
			return exactProbability(betaInterval(left, right, alpha[0], alpha[1]));
		}
		return MultivariateProbability.integrate(new MultivariateProbability.Integrand() {
			@Override public int dimension() { return alpha.length - 2; }
			@Override public double value(double[] uniforms) {
				double remaining = 1.0;
				double product = 1.0;
				for (int i = 0; i + 1 < alpha.length; i++) {
					double xLower = Math.max(lo[i], remaining - futureUpper[i + 1]);
					double xUpper = Math.min(hi[i], remaining - futureLower[i + 1]);
					if (!(xLower < xUpper) || !(remaining > 0.0)) return 0.0;
					double vLower = Math.max(0.0, xLower / remaining);
					double vUpper = Math.min(1.0, xUpper / remaining);
					double probability = betaInterval(vLower, vUpper, alpha[i],
							remainingAlpha[i] - alpha[i]);
					if (!(probability > 0.0)) return 0.0;
					product *= probability;
					if (i + 2 == alpha.length) break;
					double beta = remainingAlpha[i] - alpha[i];
					double v;
					if (vLower >= 0.5) {
						double lowerSurvival = Beta.cumulative(vLower, alpha[i], beta,
								false, false);
						double target = Math.max(Math.nextUp(0.0), Math.min(
								Math.nextAfter(1.0, 0.0), lowerSurvival -
								uniforms[i] * probability));
						v = Beta.quantile(target, alpha[i], beta, false, false);
					} else {
						double lowerCdf = Beta.cumulative(vLower, alpha[i], beta,
								true, false);
						double target = Math.max(Math.nextUp(0.0), Math.min(
								Math.nextAfter(1.0, 0.0), lowerCdf +
								uniforms[i] * probability));
						v = Beta.quantile(target, alpha[i], beta, true, false);
					}
					remaining *= 1.0 - v;
				}
				return product;
			}
		}, options, random);
	}

	public static MultivariateProbabilityResult probability(double[] lower,
			double[] upper, double[] alpha) {
		return probability(lower, upper, alpha, new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	/** Computes {@code P(X[i] <= upper[i], all i)} on the simplex. */
	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] alpha, MultivariateProbabilityOptions options,
			RandomEngine random) {
		if (upper == null) return invalidProbability();
		double[] lower = new double[upper.length];
		Arrays.fill(lower, Double.NEGATIVE_INFINITY);
		return probability(lower, upper, alpha, options, random);
	}

	public static MultivariateProbabilityResult cumulative(double[] upper,
			double[] alpha) {
		return cumulative(upper, alpha, new MultivariateProbabilityOptions(),
				new MersenneTwister(PROBABILITY_SEED));
	}

	private static double betaInterval(double lower, double upper, double a,
			double b) {
		if (!(lower < upper)) return 0.0;
		if (lower >= 0.5) return Math.max(0.0,
				Beta.cumulative(lower, a, b, false, false) -
				Beta.cumulative(upper, a, b, false, false));
		return Math.max(0.0, Beta.cumulative(upper, a, b, true, false) -
				Beta.cumulative(lower, a, b, true, false));
	}

	private static MultivariateProbabilityResult exactProbability(double value) {
		return new MultivariateProbabilityResult(value, 0.0, 0, 0);
	}

	private static MultivariateProbabilityResult invalidProbability() {
		return new MultivariateProbabilityResult(Double.NaN, Double.NaN, 0, 2);
	}
}
