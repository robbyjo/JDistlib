/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** A differentiable map from unconstrained coordinates to constrained values. */
public interface ParameterConstraint {
	int unconstrainedDimension();
	int constrainedDimension();
	String description();

	/** Constrains a slice and returns its log absolute Jacobian determinant. */
	double constrain(double[] source, int sourceOffset, double[] target,
			int targetOffset);

	/** Maps constrained values back to unconstrained coordinates. */
	void unconstrain(double[] source, int sourceOffset, double[] target,
			int targetOffset);

	/** Pulls a constrained gradient back and adds the log-Jacobian gradient. */
	void pullback(double[] unconstrained, int unconstrainedOffset,
			double[] constrained, int constrainedOffset,
			double[] constrainedGradient, double[] unconstrainedGradient);
}
