/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable parameter metadata in a compiled Bayesian model. */
public final class ParameterSpec {
	private final String name;
	private final ParameterConstraint constraint;
	private final int unconstrainedOffset;
	private final int constrainedOffset;

	ParameterSpec(String name, ParameterConstraint constraint,
			int unconstrainedOffset, int constrainedOffset) {
		this.name = name;
		this.constraint = constraint;
		this.unconstrainedOffset = unconstrainedOffset;
		this.constrainedOffset = constrainedOffset;
	}

	public String name() { return name; }
	public ParameterConstraint constraint() { return constraint; }
	public int unconstrainedOffset() { return unconstrainedOffset; }
	public int constrainedOffset() { return constrainedOffset; }
	public int unconstrainedDimension() { return constraint.unconstrainedDimension(); }
	public int constrainedDimension() { return constraint.constrainedDimension(); }
}
