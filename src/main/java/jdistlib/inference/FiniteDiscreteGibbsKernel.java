/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Exact full-conditional update for one finite integer or categorical coordinate. */
public final class FiniteDiscreteGibbsKernel implements HybridKernel {
	private final int coordinate;
	public FiniteDiscreteGibbsKernel(int coordinate) { if (coordinate < 0) throw new IllegalArgumentException("coordinate must be nonnegative"); this.coordinate = coordinate; }
	@Override public String name() { return "discrete-gibbs[" + coordinate + "]"; }
	@Override public HybridKernelTransition update(double[] state, double currentLogDensity,
			LogDensity target, MixedStateSpace space, RandomEngine random) {
		check(state, target, space, random); CoordinateSupport support = space.support(coordinate);
		if (!support.discrete() || !support.finite()) throw new IllegalArgumentException("finite discrete support required");
		long countLong = (long) support.upper() - (long) support.lower() + 1L;
		if (countLong > 1000000L) throw new IllegalArgumentException("finite Gibbs support is too large");
		int count = (int) countLong, lower = (int) support.lower(); double[] logMass = new double[count]; double maximum = Double.NEGATIVE_INFINITY;
		double previous = state[coordinate];
		for (int i = 0; i < count; i++) { state[coordinate] = lower + i; logMass[i] = target.logDensity(state); maximum = Math.max(maximum, logMass[i]); }
		state[coordinate] = previous;
		if (!Double.isFinite(maximum)) return new HybridKernelTransition(currentLogDensity, false, 0.0, false);
		double total = 0.0; for (double value : logMass) total += Math.exp(value - maximum);
		double threshold = random.nextDouble() * total, cumulative = 0.0; int selected = count - 1;
		for (int i = 0; i < count; i++) { cumulative += Math.exp(logMass[i] - maximum); if (threshold <= cumulative) { selected = i; break; } }
		state[coordinate] = lower + selected;
		return new HybridKernelTransition(logMass[selected], true, 1.0, false);
	}
	private void check(double[] state, LogDensity target, MixedStateSpace space, RandomEngine random) {
		if (state == null || target == null || space == null || random == null || coordinate >= space.dimension() || state.length != space.dimension())
			throw new IllegalArgumentException("kernel inputs do not match");
	}
}
