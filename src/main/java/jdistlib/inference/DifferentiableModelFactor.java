/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** A factor that adds derivatives with respect to all constrained coordinates. */
public interface DifferentiableModelFactor extends ModelFactor {
	double logDensityAndAddGradient(ModelState state, double[] constrainedGradient);

	@Override default double logDensity(ModelState state) {
		return logDensityAndAddGradient(state,
				new double[state.constrainedDimension()]);
	}
}
