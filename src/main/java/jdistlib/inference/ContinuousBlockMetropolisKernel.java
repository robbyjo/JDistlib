/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Gaussian random-walk update of a continuous block conditional on all other coordinates. */
public final class ContinuousBlockMetropolisKernel implements HybridKernel {
	private final int[] coordinates; private final double[] scales;
	public ContinuousBlockMetropolisKernel(int[] coordinates, double scale) {
		this(coordinates, fill(coordinates, scale));
	}
	public ContinuousBlockMetropolisKernel(int[] coordinates, double[] scales) {
		if (coordinates == null || scales == null || coordinates.length == 0 || coordinates.length != scales.length)
			throw new IllegalArgumentException("coordinates and scales must match");
		this.coordinates = coordinates.clone(); this.scales = scales.clone();
		for (int i = 0; i < this.coordinates.length; i++) {
			if (this.coordinates[i] < 0 || !(this.scales[i] > 0.0) || !Double.isFinite(this.scales[i])) throw new IllegalArgumentException("invalid coordinate or scale");
			for (int j = 0; j < i; j++) if (this.coordinates[j] == this.coordinates[i]) throw new IllegalArgumentException("coordinates must be unique");
		}
	}
	private static double[] fill(int[] coordinates, double scale) {
		if (coordinates == null) throw new IllegalArgumentException("coordinates required");
		double[] result = new double[coordinates.length]; for (int i = 0; i < result.length; i++) result[i] = scale; return result;
	}
	@Override public String name() { return "continuous-block"; }
	@Override public HybridKernelTransition update(double[] state, double currentLogDensity,
			LogDensity target, MixedStateSpace space, RandomEngine random) {
		if (state == null || target == null || space == null || random == null || state.length != space.dimension()) throw new IllegalArgumentException("kernel inputs do not match");
		double[] previous = new double[coordinates.length];
		for (int i = 0; i < coordinates.length; i++) {
			int coordinate = coordinates[i];
			if (coordinate >= space.dimension() || space.support(coordinate).discrete()) throw new IllegalArgumentException("continuous coordinates required");
			previous[i] = state[coordinate]; state[coordinate] += scales[i] * random.nextGaussian();
		}
		if (!space.contains(state)) { restore(state, previous); return new HybridKernelTransition(currentLogDensity, false, 0.0, true); }
		double proposedLogDensity = target.logDensity(state);
		double probability = Double.isFinite(proposedLogDensity) ? Math.min(1.0, Math.exp(proposedLogDensity - currentLogDensity)) : 0.0;
		if (random.nextDouble() < probability) return new HybridKernelTransition(proposedLogDensity, true, probability, false);
		restore(state, previous); return new HybridKernelTransition(currentLogDensity, false, probability, false);
	}
	private void restore(double[] state, double[] previous) { for (int i = 0; i < coordinates.length; i++) state[coordinates[i]] = previous[i]; }
}
