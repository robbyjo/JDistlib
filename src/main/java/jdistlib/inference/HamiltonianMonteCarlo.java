/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Fixed-trajectory HMC with dual-averaged step size and covariance adaptation. */
public final class HamiltonianMonteCarlo implements Sampler {
	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || initialState.length == 0
				|| options == null || random == null)
			throw new IllegalArgumentException("target, state, options and random are required");
		DifferentiableLogDensity differentiable = HamiltonianSupport.gradientTarget(target, options);
		HamiltonianSupport.Point state = HamiltonianSupport.at(differentiable, initialState);
		ChainAccumulator output = new ChainAccumulator();
		if (!finite(state)) {
			output.warn("initial log density or gradient is not finite");
			return output.result(initialState, state.logDensity, 0, random, null,
					ChainResult.Status.INVALID_INITIAL_STATE);
		}
		HamiltonianSupport.MassMatrix mass = new HamiltonianSupport.MassMatrix(initialState.length);
		double initialStep = options.adaptStepSize()
				? HamiltonianSupport.findReasonableStep(differentiable,
						state, mass, random, options.stepSize()) : options.stepSize();
		double step = initialStep;
		HamiltonianSupport.DualAveraging adaptation =
				new HamiltonianSupport.DualAveraging(step, options.targetAcceptance());
		HamiltonianSupport.RunningCovariance covariance =
				new HamiltonianSupport.RunningCovariance(initialState.length);
		double acceptanceSum = 0.0;
		int completed = 0;
		int total = options.warmupIterations()
				+ options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return result(output, state, completed, random,
					options, initialStep, step, mass, acceptanceSum, ChainResult.Status.CANCELLED);
			double[] momentum = mass.momentum(random);
			HamiltonianSupport.Point proposal = new HamiltonianSupport.Point(state.q,
					momentum, state.gradient, state.logDensity);
			double initialEnergy = -state.logDensity + mass.kinetic(momentum);
			for (int leapfrog = 0; leapfrog < options.leapfrogSteps(); leapfrog++)
				proposal = HamiltonianSupport.leapfrog(differentiable, proposal, step, mass);
			double proposedEnergy = -proposal.logDensity + mass.kinetic(proposal.p);
			double error = proposedEnergy - initialEnergy;
			double probability = finite(proposal) && Double.isFinite(error)
					? Math.min(1.0, Math.exp(-error)) : 0.0;
			boolean divergent = !Double.isFinite(error)
					|| Math.abs(error) > options.maximumEnergyError();
			boolean accepted = !divergent && random.nextDouble() < probability;
			if (accepted) state = new HamiltonianSupport.Point(proposal.q,
					new double[proposal.q.length], proposal.gradient, proposal.logDensity);
			completed++;
			if (iteration < options.warmupIterations()) {
				acceptanceSum += probability;
				covariance.add(state.q);
				if (options.adaptStepSize()) step = adaptation.update(probability);
				if (options.adaptMassMatrix() && iteration > 50 && (iteration + 1) % 50 == 0) {
					mass.update(covariance.covariance(), options.denseMassMatrix());
				}
				if (iteration + 1 == options.warmupIterations() && options.adaptStepSize())
					step = adaptation.averaged();
			} else if ((iteration - options.warmupIterations() + 1)
					% options.thinning() == 0) {
				output.add(state.q, state.logDensity, new IterationStats(accepted,
						probability, step, proposedEnergy, error, divergent, 0,
						options.leapfrogSteps()));
			}
		}
		return result(output, state, completed, random, options, initialStep, step,
				mass, acceptanceSum, ChainResult.Status.SUCCESS);
	}

	private static boolean finite(HamiltonianSupport.Point point) {
		if (!Double.isFinite(point.logDensity)) return false;
		for (double value : point.gradient) if (!Double.isFinite(value)) return false;
		return true;
	}
	private static ChainResult result(ChainAccumulator output,
			HamiltonianSupport.Point state, int completed, RandomEngine random,
			SamplingOptions options, double initialStep, double finalStep,
			HamiltonianSupport.MassMatrix mass, double acceptanceSum,
			ChainResult.Status status) {
		WarmupResult warmup = WarmupResult.withInverseMassMatrix(
				options.warmupIterations(), initialStep,
				finalStep, mass.inverseMatrix(), options.warmupIterations() == 0
				? Double.NaN : acceptanceSum / options.warmupIterations());
		return output.result(state.q, state.logDensity, completed, random, warmup, status);
	}
}
