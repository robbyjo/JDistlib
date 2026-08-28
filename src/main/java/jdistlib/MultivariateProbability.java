/*
 * Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later
 *
 * Uses Alan Genz's sequential-conditioning transformation together with a
 * randomized, antithetic Halton rule. Integration state and random shifts are
 * local to each call.
 */
package jdistlib;

import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Internal numerical core for multivariate normal and central-t probabilities. */
final class MultivariateProbability {
	private static final double ERROR_MULTIPLIER = 3.5;
	private static final double MIN_PROBABILITY = Math.nextUp(0.0);
	private static final double MAX_PROBABILITY = Math.nextAfter(1.0, 0.0);

	private MultivariateProbability() {}

	interface Integrand {
		int dimension();
		double value(double[] uniforms);
	}

	static MultivariateProbabilityResult normal(double[] lower, double[] upper,
			double[] mean, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		Prepared prepared = prepare(lower, upper, mean, covariance, false, 0.0);
		if (prepared.invalid) return invalid();
		if (prepared.empty) return exact(0.0);
		if (prepared.unrestricted) return exact(1.0);
		if (prepared.mean.length == 1) {
			double scale = prepared.factor.lower[0][0];
			return exact(normalInterval((prepared.lower[0] - prepared.mean[0]) / scale,
					(prepared.upper[0] - prepared.mean[0]) / scale));
		}
		return integrate(new NormalIntegrand(prepared), options, random);
	}

	static MultivariateProbabilityResult studentT(double[] lower, double[] upper,
			double[] location, double[][] scale, double degreesOfFreedom,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom))
			return invalid();
		Prepared prepared = prepare(lower, upper, location, scale, true,
				degreesOfFreedom);
		if (prepared.invalid) return invalid();
		if (prepared.empty) return exact(0.0);
		if (prepared.unrestricted) return exact(1.0);
		if (prepared.mean.length == 1) {
			double scaleValue = prepared.factor.lower[0][0];
			double probability = scalarTInterval(
					(prepared.lower[0] - prepared.mean[0]) / scaleValue,
					(prepared.upper[0] - prepared.mean[0]) / scaleValue,
					degreesOfFreedom);
			return exact(probability);
		}
		return integrate(new StudentTIntegrand(prepared, degreesOfFreedom), options,
				random);
	}

	static MultivariateProbabilityResult laplace(double[] lower, double[] upper,
			double[] location, double[][] covariance,
			MultivariateProbabilityOptions options, RandomEngine random) {
		Prepared prepared = prepare(lower, upper, location, covariance, false, 0.0);
		if (prepared.invalid) return invalid();
		if (prepared.empty) return exact(0.0);
		if (prepared.unrestricted) return exact(1.0);
		if (prepared.mean.length == 1) {
			double scale = prepared.factor.lower[0][0] / Math.sqrt(2.0);
			double probability = Laplace.cumulative(prepared.upper[0],
					prepared.mean[0], scale, true, false) -
					Laplace.cumulative(prepared.lower[0], prepared.mean[0], scale,
							true, false);
			return exact(Math.max(0.0, probability));
		}
		return integrate(new LaplaceIntegrand(prepared), options, random);
	}

	static MultivariateProbabilityResult integrate(Integrand integrand,
			MultivariateProbabilityOptions options, RandomEngine random) {
		if (options == null || !options.isValid() || random == null) return invalid();
		int replications = options.replications;
		int maximumPoints = options.maxEvaluations / (2 * replications);
		if (maximumPoints < 1) return invalid();
		int dimensions = integrand.dimension();
		int[] primes = primes(dimensions);
		double[][] shifts = new double[replications][dimensions];
		for (int replication = 0; replication < replications; replication++)
			for (int dimension = 0; dimension < dimensions; dimension++)
				shifts[replication][dimension] = unitOpen(random.nextDouble());

		double[] sums = new double[replications];
		int completedPoints = 0;
		int targetPoints = Math.min(32, maximumPoints);
		double probability = Double.NaN;
		double error = Double.POSITIVE_INFINITY;
		int evaluations = 0;
		while (completedPoints < maximumPoints) {
			for (int replication = 0; replication < replications; replication++) {
				double[] point = new double[dimensions];
				double[] antithetic = new double[dimensions];
				for (int index = completedPoints + 1; index <= targetPoints; index++) {
					for (int dimension = 0; dimension < dimensions; dimension++) {
						double shifted = radicalInverse(index, primes[dimension]) +
								shifts[replication][dimension];
						shifted -= Math.floor(shifted);
						point[dimension] = unitOpen(shifted);
						antithetic[dimension] = unitOpen(1.0 - shifted);
					}
					sums[replication] += integrand.value(point) +
							integrand.value(antithetic);
				}
			}
			completedPoints = targetPoints;
			evaluations = 2 * replications * completedPoints;
			double[] estimates = new double[replications];
			probability = 0.0;
			for (int replication = 0; replication < replications; replication++) {
				estimates[replication] = sums[replication] / (2.0 * completedPoints);
				probability += estimates[replication];
			}
			probability /= replications;
			double variance = 0.0;
			for (double estimate : estimates) {
				double difference = estimate - probability;
				variance += difference * difference;
			}
			variance /= replications - 1.0;
			error = ERROR_MULTIPLIER * Math.sqrt(variance / replications);
			double tolerance = options.toleranceFor(probability);
			if (error <= tolerance) {
				return result(probability, error, evaluations, 0);
			}
			if (completedPoints == maximumPoints) break;
			targetPoints = Math.min(maximumPoints, completedPoints * 2);
		}
		return result(probability, error, evaluations, 1);
	}

	private static Prepared prepare(double[] lower, double[] upper, double[] mean,
			double[][] covariance, boolean studentT, double degreesOfFreedom) {
		if (lower == null || upper == null || mean == null || covariance == null ||
				mean.length == 0 || lower.length != mean.length ||
				upper.length != mean.length || covariance.length != mean.length)
			return Prepared.invalid();
		int dimension = mean.length;
		for (int i = 0; i < dimension; i++) {
			if (Double.isNaN(lower[i]) || Double.isNaN(upper[i]) ||
					!Double.isFinite(mean[i])) return Prepared.invalid();
			if (lower[i] > upper[i]) return Prepared.empty();
			if (lower[i] == upper[i]) return Prepared.empty();
		}
		MultivariateDistributionUtil.Factor original =
				MultivariateDistributionUtil.factor(mean, covariance);
		if (original == null) return Prepared.invalid();
		boolean unrestricted = true;
		for (int i = 0; i < dimension; i++)
			unrestricted &= lower[i] == Double.NEGATIVE_INFINITY &&
					upper[i] == Double.POSITIVE_INFINITY;
		if (unrestricted) return Prepared.unrestricted(mean.length);

		Integer[] order = new Integer[dimension];
		for (int i = 0; i < dimension; i++) order[i] = Integer.valueOf(i);
		Arrays.sort(order, (left, right) -> {
			double leftScale = Math.sqrt(covariance[left.intValue()][left.intValue()]);
			double rightScale = Math.sqrt(covariance[right.intValue()][right.intValue()]);
			double leftLower = (lower[left.intValue()] - mean[left.intValue()]) /
					leftScale;
			double leftUpper = (upper[left.intValue()] - mean[left.intValue()]) /
					leftScale;
			double rightLower = (lower[right.intValue()] - mean[right.intValue()]) /
					rightScale;
			double rightUpper = (upper[right.intValue()] - mean[right.intValue()]) /
					rightScale;
			double leftProbability = studentT
					? scalarTInterval(leftLower, leftUpper, degreesOfFreedom)
					: normalInterval(leftLower, leftUpper);
			double rightProbability = studentT
					? scalarTInterval(rightLower, rightUpper, degreesOfFreedom)
					: normalInterval(rightLower, rightUpper);
			return Double.compare(leftProbability, rightProbability);
		});
		double[] sortedLower = new double[dimension];
		double[] sortedUpper = new double[dimension];
		double[] sortedMean = new double[dimension];
		double[][] sortedCovariance = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) {
			int sourceI = order[i].intValue();
			sortedLower[i] = lower[sourceI];
			sortedUpper[i] = upper[sourceI];
			sortedMean[i] = mean[sourceI];
			for (int j = 0; j < dimension; j++)
				sortedCovariance[i][j] =
						covariance[sourceI][order[j].intValue()];
		}
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(sortedMean, sortedCovariance);
		if (factor == null) return Prepared.invalid();
		return new Prepared(sortedLower, sortedUpper, sortedMean, factor, false,
				false, false);
	}

	private static final class NormalIntegrand implements Integrand {
		private final Prepared prepared;

		NormalIntegrand(Prepared prepared) {
			this.prepared = prepared;
		}

		@Override public int dimension() {
			return prepared.mean.length - 1;
		}

		@Override public double value(double[] uniforms) {
			return conditionalNormal(prepared, 1.0, uniforms, 0);
		}
	}

	private static final class StudentTIntegrand implements Integrand {
		private final Prepared prepared;
		private final double degreesOfFreedom;

		StudentTIntegrand(Prepared prepared, double degreesOfFreedom) {
			this.prepared = prepared;
			this.degreesOfFreedom = degreesOfFreedom;
		}

		@Override public int dimension() {
			return prepared.mean.length;
		}

		@Override public double value(double[] uniforms) {
			double chiSquare = ChiSquare.quantile(unitOpen(uniforms[0]),
					degreesOfFreedom, true, false);
			double scale = Math.sqrt(chiSquare / degreesOfFreedom);
			return conditionalNormal(prepared, scale, uniforms, 1);
		}
	}

	private static final class LaplaceIntegrand implements Integrand {
		private final Prepared prepared;

		LaplaceIntegrand(Prepared prepared) { this.prepared = prepared; }

		@Override public int dimension() { return prepared.mean.length; }

		@Override public double value(double[] uniforms) {
			double mixing = -Math.log1p(-unitOpen(uniforms[0]));
			return conditionalNormal(prepared, 1.0 / Math.sqrt(mixing), uniforms, 1);
		}
	}

	private static double conditionalNormal(Prepared prepared, double boundScale,
			double[] uniforms, int uniformOffset) {
		int dimension = prepared.mean.length;
		double[] latent = new double[dimension];
		double product = 1.0;
		for (int i = 0; i < dimension; i++) {
			double conditionalMean = prepared.mean[i];
			for (int j = 0; j < i; j++)
				conditionalMean += prepared.factor.lower[i][j] * latent[j];
			double diagonal = prepared.factor.lower[i][i];
			double standardizedLower = boundScale *
					(prepared.lower[i] - prepared.mean[i]) / diagonal;
			double standardizedUpper = boundScale *
					(prepared.upper[i] - prepared.mean[i]) / diagonal;
			if (i > 0) {
				double shift = (conditionalMean - prepared.mean[i]) / diagonal;
				standardizedLower -= shift;
				standardizedUpper -= shift;
			}
			double[] sampled = new double[1];
			double probability = truncatedNormal(standardizedLower,
					standardizedUpper, i + 1 < dimension
							? uniforms[uniformOffset + i] : 0.5, sampled);
			if (!(probability > 0.0)) return 0.0;
			product *= probability;
			if (i + 1 < dimension) latent[i] = sampled[0];
		}
		return product;
	}

	private static double truncatedNormal(double lower, double upper, double u,
			double[] sampled) {
		if (lower >= upper) return 0.0;
		u = unitOpen(u);
		double probability;
		if (lower >= 0.0) {
			double lowerSurvival = Normal.cumulative(lower, 0.0, 1.0, false, false);
			double upperSurvival = Normal.cumulative(upper, 0.0, 1.0, false, false);
			probability = lowerSurvival - upperSurvival;
			if (probability > 0.0) {
				double target = unitOpen(lowerSurvival - u * probability);
				sampled[0] = Normal.quantile(target, 0.0, 1.0, false, false);
			}
		} else {
			double lowerCdf = Normal.cumulative(lower, 0.0, 1.0, true, false);
			double upperCdf = Normal.cumulative(upper, 0.0, 1.0, true, false);
			probability = upperCdf - lowerCdf;
			if (probability > 0.0) {
				double target = unitOpen(lowerCdf + u * probability);
				sampled[0] = Normal.quantile(target, 0.0, 1.0, true, false);
			}
		}
		return probability;
	}

	private static double normalInterval(double lower, double upper) {
		if (lower >= upper) return 0.0;
		if (lower >= 0.0) {
			return Math.max(0.0, Normal.cumulative(lower, 0.0, 1.0, false, false) -
					Normal.cumulative(upper, 0.0, 1.0, false, false));
		}
		return Math.max(0.0, Normal.cumulative(upper, 0.0, 1.0, true, false) -
				Normal.cumulative(lower, 0.0, 1.0, true, false));
	}

	private static double scalarTInterval(double lower, double upper,
			double degreesOfFreedom) {
		if (lower >= upper) return 0.0;
		if (lower >= 0.0) {
			return Math.max(0.0, T.cumulative(lower, degreesOfFreedom, false, false) -
					T.cumulative(upper, degreesOfFreedom, false, false));
		}
		return Math.max(0.0, T.cumulative(upper, degreesOfFreedom, true, false) -
				T.cumulative(lower, degreesOfFreedom, true, false));
	}

	private static int[] primes(int count) {
		int[] result = new int[count];
		int candidate = 2;
		for (int found = 0; found < count; candidate++) {
			boolean prime = true;
			for (int divisor = 2; divisor * divisor <= candidate; divisor++) {
				if (candidate % divisor == 0) {
					prime = false;
					break;
				}
			}
			if (prime) result[found++] = candidate;
		}
		return result;
	}

	private static double radicalInverse(int index, int base) {
		double inverse = 1.0 / base;
		double factor = inverse;
		double result = 0.0;
		while (index > 0) {
			result += (index % base) * factor;
			index /= base;
			factor *= inverse;
		}
		return result;
	}

	private static double unitOpen(double value) {
		if (!(value > 0.0)) return MIN_PROBABILITY;
		if (!(value < 1.0)) return MAX_PROBABILITY;
		return value;
	}

	private static MultivariateProbabilityResult exact(double probability) {
		return new MultivariateProbabilityResult(probability, 0.0, 0, 0);
	}

	private static MultivariateProbabilityResult invalid() {
		return new MultivariateProbabilityResult(Double.NaN, Double.NaN, 0, 2);
	}

	private static MultivariateProbabilityResult result(double probability,
			double error, int evaluations, int status) {
		return new MultivariateProbabilityResult(Math.max(0.0, Math.min(1.0,
				probability)), error, evaluations, status);
	}

	private static final class Prepared {
		final double[] lower;
		final double[] upper;
		final double[] mean;
		final MultivariateDistributionUtil.Factor factor;
		final boolean invalid;
		final boolean empty;
		final boolean unrestricted;

		Prepared(double[] lower, double[] upper, double[] mean,
				MultivariateDistributionUtil.Factor factor, boolean invalid,
				boolean empty, boolean unrestricted) {
			this.lower = lower;
			this.upper = upper;
			this.mean = mean;
			this.factor = factor;
			this.invalid = invalid;
			this.empty = empty;
			this.unrestricted = unrestricted;
		}

		static Prepared invalid() {
			return new Prepared(null, null, null, null, true, false, false);
		}

		static Prepared empty() {
			return new Prepared(null, null, null, null, false, true, false);
		}

		static Prepared unrestricted(int dimension) {
			return new Prepared(null, null, new double[dimension], null, false,
					false, true);
		}
	}
}
