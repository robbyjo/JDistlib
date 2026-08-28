/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/**
 * Metropolis-adjusted microcanonical Langevin sampler (MHMCHMC).
 *
 * <p>This is the adjusted, asymptotically exact counterpart of MCLMC.  It uses
 * the reversible isokinetic McLachlan splitting and a unit-norm momentum.  The
 * method requires at least two dimensions.</p>
 */
public final class AdjustedMicrocanonicalLangevin implements Sampler {
	private static final double B1 = 0.1931833275037836;
	private static final double B2 = 1.0 - 2.0 * B1;

	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || options == null || random == null)
			throw new IllegalArgumentException("target, state, options and random are required");
		if (initialState.length < 2)
			throw new IllegalArgumentException("adjusted MCLMC requires dimension >= 2");
		DifferentiableLogDensity density = HamiltonianSupport.gradientTarget(target, options);
		double[] q = initialState.clone();
		double[] gradient = new double[q.length];
		double value = density.logDensityAndGradient(q, gradient);
		ChainAccumulator output = new ChainAccumulator();
		if (!Double.isFinite(value))
			return output.result(q, value, 0, random, null,
					ChainResult.Status.INVALID_INITIAL_STATE);
		double step = options.stepSize();
		double initialStep = step;
		double acceptanceSum = 0.0;
		int warmup = options.warmupIterations();
		int total = warmup + options.sampleIterations() * options.thinning();
		int retained = 0;
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled())
				return output.result(q, value, retained, random, warmupResult(warmup,
						initialStep, step, acceptanceSum), ChainResult.Status.CANCELLED);
			double actualStep = HamiltonianSupport.jitter(step,
					options.stepSizeJitter(), random);
			Transition transition = transition(density, q, gradient, value, actualStep,
					options.leapfrogSteps(), options.maximumEnergyError(), random);
			if (random.nextDouble() < transition.acceptanceProbability) {
				q = transition.position; gradient = transition.gradient;
				value = transition.logDensity;
			}
			boolean accepted = q == transition.position;
			IterationStats stats = new IterationStats(accepted,
					transition.acceptanceProbability, actualStep, -value,
					transition.energyError, transition.divergent, 0,
					options.leapfrogSteps());
			if (iteration < warmup) {
				acceptanceSum += transition.acceptanceProbability;
				if (options.adaptStepSize()) {
					double gain = 1.0 / Math.sqrt(iteration + 10.0);
					step *= Math.exp(gain * (transition.acceptanceProbability
							- options.targetAcceptance()));
					step = Math.max(1e-8, Math.min(10.0, step));
				}
				options.progress(iteration + 1, total, true, stats);
			} else {
				int postWarmup = iteration - warmup;
				if ((postWarmup + 1) % options.thinning() == 0) {
					output.retain(options, q, value, stats); retained++;
				}
				options.progress(iteration + 1, total, false, stats);
			}
		}
		return output.result(q, value, retained, random, warmupResult(warmup,
				initialStep, step, acceptanceSum), ChainResult.Status.SUCCESS);
	}

	private static Transition transition(DifferentiableLogDensity density,
			double[] position, double[] oldGradient, double oldValue, double step,
			int steps, double maximumEnergyError, RandomEngine random) {
		int dimension = position.length;
		double[] q = position.clone();
		double[] p = unitGaussian(dimension, random);
		double[] gradient = oldGradient.clone();
		double kineticChange = 0.0;
		for (int i = 0; i < steps; i++) {
			kineticChange += momentumUpdate(p, gradient, step * B1, dimension);
			positionUpdate(q, p, step * 0.5);
			double value = density.logDensityAndGradient(q, gradient);
			if (!Double.isFinite(value))
				return new Transition(q, gradient, value, 0.0,
						Double.POSITIVE_INFINITY, true);
			kineticChange += momentumUpdate(p, gradient, step * B2, dimension);
			positionUpdate(q, p, step * 0.5);
			value = density.logDensityAndGradient(q, gradient);
			if (!Double.isFinite(value))
				return new Transition(q, gradient, value, 0.0,
						Double.POSITIVE_INFINITY, true);
			kineticChange += momentumUpdate(p, gradient, step * B1, dimension);
		}
		double value = density.logDensityAndGradient(q, gradient);
		double logRatio = value - oldValue - kineticChange;
		double probability = Double.isFinite(logRatio)
				? Math.min(1.0, Math.exp(Math.min(0.0, logRatio))) : 0.0;
		double error = -logRatio;
		return new Transition(q, gradient, value, probability, error,
				!Double.isFinite(error) || error > maximumEnergyError);
	}

	private static double momentumUpdate(double[] momentum, double[] gradient,
			double scale, int dimension) {
		double norm = norm(gradient);
		if (!(norm > 0.0) || !Double.isFinite(norm)) return 0.0;
		double projection = 0.0;
		for (int i = 0; i < dimension; i++) projection += momentum[i] * gradient[i] / norm;
		double delta = scale * norm / (dimension - 1.0);
		double zeta = Math.exp(Math.max(-350.0, Math.min(350.0, -delta)));
		double[] raw = new double[dimension];
		double rawNorm = 0.0;
		for (int i = 0; i < dimension; i++) {
			double direction = gradient[i] / norm;
			raw[i] = direction * (1.0 - zeta)
					* (1.0 + zeta + projection * (1.0 - zeta))
					+ 2.0 * zeta * momentum[i];
			rawNorm += raw[i] * raw[i];
		}
		rawNorm = Math.sqrt(rawNorm);
		for (int i = 0; i < dimension; i++) momentum[i] = raw[i] / rawNorm;
		double term = 1.0 + projection + (1.0 - projection) * zeta * zeta;
		return (delta - Math.log(2.0) + Math.log(term)) * (dimension - 1.0);
	}

	private static void positionUpdate(double[] position, double[] momentum, double scale) {
		for (int i = 0; i < position.length; i++) position[i] += scale * momentum[i];
	}
	private static double[] unitGaussian(int dimension, RandomEngine random) {
		double[] result = new double[dimension];
		double norm;
		do {
			norm = 0.0;
			for (int i = 0; i < dimension; i++) {
				result[i] = random.nextGaussian(); norm += result[i] * result[i];
			}
			norm = Math.sqrt(norm);
		} while (!(norm > 0.0));
		for (int i = 0; i < dimension; i++) result[i] /= norm;
		return result;
	}
	private static double norm(double[] values) {
		double result = 0.0;
		for (double value : values) result += value * value;
		return Math.sqrt(result);
	}
	private static WarmupResult warmupResult(int iterations, double initial,
			double step, double sum) {
		return new WarmupResult(iterations, initial, step, null,
				iterations == 0 ? Double.NaN : sum / iterations);
	}
	private static final class Transition {
		final double[] position, gradient;
		final double logDensity, acceptanceProbability, energyError;
		final boolean divergent;
		Transition(double[] position, double[] gradient, double logDensity,
				double acceptanceProbability, double energyError, boolean divergent) {
			this.position = position; this.gradient = gradient; this.logDensity = logDensity;
			this.acceptanceProbability = acceptanceProbability;
			this.energyError = energyError; this.divergent = divergent;
		}
	}
}
