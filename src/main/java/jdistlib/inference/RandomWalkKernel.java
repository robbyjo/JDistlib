/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Reusable Gaussian random-walk Metropolis transition kernel. */
public final class RandomWalkKernel implements TransitionKernel<RandomWalkKernel.State> {
	public static final class State {
		private final double[] position;
		private final double logDensity;
		State(double[] position, double logDensity) {
			this.position = position.clone(); this.logDensity = logDensity;
		}
		public double[] position() { return position.clone(); }
		public double logDensity() { return logDensity; }
	}
	@Override public State initialize(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || initialState.length == 0)
			throw new IllegalArgumentException("target and nonempty state are required");
		double value = target.logDensity(initialState);
		if (!Double.isFinite(value)) throw new IllegalArgumentException("initial density is not finite");
		return new State(initialState, value);
	}
	@Override public KernelTransition<State> step(LogDensity target, State state,
			SamplingOptions options, RandomEngine random) {
		if (target == null || state == null || options == null || random == null)
			throw new IllegalArgumentException("kernel arguments are required");
		double[] proposal = state.position();
		for (int i = 0; i < proposal.length; i++) proposal[i] += options.stepSize() * random.nextGaussian();
		double proposed = target.logDensity(proposal);
		double probability = Double.isFinite(proposed)
				? Math.min(1.0, Math.exp(Math.min(0.0, proposed - state.logDensity))) : 0.0;
		boolean accepted = random.nextDouble() < probability;
		State next = accepted ? new State(proposal, proposed) : state;
		IterationStats stats = new IterationStats(accepted, probability,
				options.stepSize(), -next.logDensity, Double.NaN, false, 0, 0);
		return new KernelTransition<State>(next, next.position, next.logDensity, stats);
	}
}
