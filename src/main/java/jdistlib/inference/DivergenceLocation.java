/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Location and energy error for one divergent retained transition. */
public final class DivergenceLocation {
	private final int draw;
	private final double[] unconstrained, constrained;
	private final double energyError;
	DivergenceLocation(int draw, double[] unconstrained, double[] constrained, double energyError) {
		this.draw = draw; this.unconstrained = unconstrained.clone();
		this.constrained = constrained == null ? null : constrained.clone();
		this.energyError = energyError;
	}
	public int draw() { return draw; }
	public double[] unconstrained() { return unconstrained.clone(); }
	public double[] constrained() { return constrained == null ? null : constrained.clone(); }
	public double energyError() { return energyError; }
}
