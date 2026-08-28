/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Symmetric Metropolis update for one bounded or unbounded discrete coordinate. */
public final class DiscreteMetropolisKernel implements HybridKernel {
	private final int coordinate;
	public DiscreteMetropolisKernel(int coordinate) { if (coordinate < 0) throw new IllegalArgumentException("coordinate must be nonnegative"); this.coordinate = coordinate; }
	@Override public String name() { return "discrete-metropolis[" + coordinate + "]"; }
	@Override public HybridKernelTransition update(double[] state, double currentLogDensity,
			LogDensity target, MixedStateSpace space, RandomEngine random) {
		if (state == null || target == null || space == null || random == null || coordinate >= space.dimension() || state.length != space.dimension())
			throw new IllegalArgumentException("kernel inputs do not match");
		CoordinateSupport support = space.support(coordinate);
		if (!support.discrete()) throw new IllegalArgumentException("discrete support required");
		double previous = state[coordinate], proposal;
		if (support.finite()) {
			long countLong = (long) support.upper() - (long) support.lower() + 1L;
			if (countLong > Integer.MAX_VALUE) throw new IllegalArgumentException("finite discrete support is too large");
			int count = (int) countLong;
			if (count < 2) return new HybridKernelTransition(currentLogDensity, false, 1.0, false);
			int previousOffset = (int) (previous - support.lower()); int selected = random.nextInt(count - 1);
			if (selected >= previousOffset) selected++; proposal = support.lower() + selected;
		} else proposal = previous + (random.nextDouble() < 0.5 ? -1.0 : 1.0);
		state[coordinate] = proposal;
		if (!space.contains(state)) { state[coordinate] = previous; return new HybridKernelTransition(currentLogDensity, false, 0.0, true); }
		double proposedLogDensity = target.logDensity(state);
		double probability = Double.isFinite(proposedLogDensity) ? Math.min(1.0, Math.exp(proposedLogDensity - currentLogDensity)) : 0.0;
		if (random.nextDouble() < probability) return new HybridKernelTransition(proposedLogDensity, true, probability, false);
		state[coordinate] = previous; return new HybridKernelTransition(currentLogDensity, false, probability, false);
	}
}
