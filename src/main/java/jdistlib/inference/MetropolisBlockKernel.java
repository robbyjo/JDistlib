/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Gaussian random-walk update for a declared continuous or discrete state block. */
public final class MetropolisBlockKernel implements GibbsKernel {
	private final int[] coordinates;
	private final double stepSize;
	private final boolean integer;
	public MetropolisBlockKernel(int[] coordinates, double stepSize, boolean integer) {
		if (coordinates == null || coordinates.length == 0 || !(stepSize > 0.0))
			throw new IllegalArgumentException("coordinates and positive step size are required");
		this.coordinates = coordinates.clone(); this.stepSize = stepSize; this.integer = integer;
	}
	@Override public void update(double[] state, LogDensity target, RandomEngine random) {
		double oldValue = target.logDensity(state);
		double[] old = new double[coordinates.length];
		for (int i = 0; i < coordinates.length; i++) {
			int coordinate = coordinates[i];
			if (coordinate < 0 || coordinate >= state.length)
				throw new IllegalArgumentException("block coordinate out of range");
			old[i] = state[coordinate];
			double change = stepSize * random.nextGaussian();
			state[coordinate] += integer ? Math.rint(change) : change;
		}
		double proposed = target.logDensity(state);
		if (!Double.isFinite(proposed)
				|| Math.log(random.nextDouble()) > proposed - oldValue) {
			for (int i = 0; i < coordinates.length; i++) state[coordinates[i]] = old[i];
		}
	}
}
