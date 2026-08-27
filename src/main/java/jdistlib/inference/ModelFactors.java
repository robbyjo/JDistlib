/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import static jdistlib.math.MathFunctions.lgammafn;

/** Analytic common priors and likelihood factors for the programmatic builder. */
public final class ModelFactors {
	private ModelFactors() {}

	public static DifferentiableModelFactor normalPrior(final String parameter,
			final double mean, final double standardDeviation) {
		if (!(standardDeviation > 0.0) || !Double.isFinite(mean)
				|| !Double.isFinite(standardDeviation))
			throw new IllegalArgumentException("finite mean and positive standard deviation are required");
		return new DifferentiableModelFactor() {
			@Override public double logDensityAndAddGradient(ModelState state, double[] gradient) {
				double x = state.scalar(parameter);
				double z = (x - mean) / standardDeviation;
				state.addGradient(parameter, 0, -z / standardDeviation, gradient);
				return -Math.log(standardDeviation) - 0.5 * Math.log(2.0 * Math.PI)
						- 0.5 * z * z;
			}
		};
	}

	public static DifferentiableModelFactor betaPrior(final String parameter,
			final double alpha, final double beta) {
		if (!(alpha > 0.0) || !(beta > 0.0))
			throw new IllegalArgumentException("positive beta shapes are required");
		return new DifferentiableModelFactor() {
			@Override public double logDensityAndAddGradient(ModelState state, double[] gradient) {
				double x = state.scalar(parameter);
				if (!(x > 0.0 && x < 1.0)) return Double.NEGATIVE_INFINITY;
				state.addGradient(parameter, 0, (alpha - 1.0) / x
						- (beta - 1.0) / (1.0 - x), gradient);
				return (alpha - 1.0) * Math.log(x) + (beta - 1.0) * Math.log1p(-x)
						+ lgammafn(alpha + beta) - lgammafn(alpha) - lgammafn(beta);
			}
		};
	}

	public static DifferentiableModelFactor normalObservations(final String data,
			final String meanParameter, final double standardDeviation) {
		if (!(standardDeviation > 0.0) || !Double.isFinite(standardDeviation))
			throw new IllegalArgumentException("positive finite standard deviation is required");
		return new DifferentiableModelFactor() {
			@Override public double logDensityAndAddGradient(ModelState state, double[] gradient) {
				double mean = state.scalar(meanParameter);
				double inverseVariance = 1.0 / (standardDeviation * standardDeviation);
				double result = 0.0;
				double derivative = 0.0;
				for (double observation : state.data().values(data)) {
					double difference = observation - mean;
					result += -Math.log(standardDeviation) - 0.5 * Math.log(2.0 * Math.PI)
							- 0.5 * difference * difference * inverseVariance;
					derivative += difference * inverseVariance;
				}
				state.addGradient(meanParameter, 0, derivative, gradient);
				return result;
			}
		};
	}

	public static DifferentiableModelFactor binomialObservation(final String successesData,
			final String trialsData, final String probabilityParameter) {
		return new DifferentiableModelFactor() {
			@Override public double logDensityAndAddGradient(ModelState state, double[] gradient) {
				double successes = state.data().scalar(successesData);
				double trials = state.data().scalar(trialsData);
				double probability = state.scalar(probabilityParameter);
				if (!(probability > 0.0 && probability < 1.0)) return Double.NEGATIVE_INFINITY;
				state.addGradient(probabilityParameter, 0, successes / probability
						- (trials - successes) / (1.0 - probability), gradient);
				return lgammafn(trials + 1.0) - lgammafn(successes + 1.0)
						- lgammafn(trials - successes + 1.0)
						+ successes * Math.log(probability)
						+ (trials - successes) * Math.log1p(-probability);
			}
		};
	}
}
