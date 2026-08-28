/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Optional accelerator-facing contract for evaluating independent states in one call. */
public interface BatchedDifferentiableLogDensity extends DifferentiableLogDensity {
	/** Replaces every output row; implementations may execute the batch concurrently. */
	default void logDensityAndGradientBatch(double[][] states, double[] logDensities,
			double[][] gradients) {
		if (states == null || logDensities == null || gradients == null
				|| states.length != logDensities.length || states.length != gradients.length)
			throw new IllegalArgumentException("batch outputs must match states");
		for (int i = 0; i < states.length; i++) {
			if (gradients[i] == null || gradients[i].length != states[i].length)
				throw new IllegalArgumentException("gradient row dimension mismatch");
			logDensities[i] = logDensityAndGradient(states[i], gradients[i]);
		}
	}
}
