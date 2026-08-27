/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.autodiff;

import jdistlib.inference.DifferentiableLogDensity;
import jdistlib.inference.GradientProvider;

/**
 * Reusable reverse-mode log density suitable for HMC and NUTS.
 *
 * <p>The primitive tape arena is reset, not reallocated, between evaluations.
 * Instances are deliberately not thread-safe; create one per sampler chain.</p>
 */
public final class ReverseModeLogDensity implements DifferentiableLogDensity, GradientProvider {
	private final int dimension;
	private final ReverseDifferentiableFunction function;
	private final ReverseModeGradient evaluator;

	public ReverseModeLogDensity(int dimension, ReverseDifferentiableFunction function) {
		this(dimension, function, Math.max(1024, dimension * 16));
	}

	public ReverseModeLogDensity(int dimension, ReverseDifferentiableFunction function,
			int initialTapeCapacity) {
		if (dimension < 1 || function == null) throw new IllegalArgumentException("positive dimension and function required");
		this.dimension = dimension; this.function = function;
		evaluator = new ReverseModeGradient(new ReverseTape(initialTapeCapacity));
	}

	public int dimension() { return dimension; }
	public ReverseTape tape() { return evaluator.tape(); }

	@Override public double logDensityAndGradient(double[] state, double[] gradient) {
		if (state == null || gradient == null || state.length != dimension || gradient.length != dimension)
			throw new IllegalArgumentException("state and gradient must match the configured dimension");
		return evaluator.evaluate(function, state, gradient);
	}

	@Override public boolean hasAnalyticGradient() { return true; }
}
