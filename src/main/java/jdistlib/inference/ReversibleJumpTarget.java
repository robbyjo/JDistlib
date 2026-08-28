/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Complete normalized joint density and schemas across a family of models. */
public interface ReversibleJumpTarget {
	ReversibleJumpModelSpace modelSpace(long modelId);
	double logJoint(ReversibleJumpState state);
	/** Fixed-model view used to reuse ordinary JDistlib kernels; override to preserve analytic gradients. */
	default LogDensity fixedModelTarget(final long modelId) {
		return new LogDensity() {
			@Override public double logDensity(double[] parameters) {
				return logJoint(new ReversibleJumpState(modelId, parameters));
			}
		};
	}
}
