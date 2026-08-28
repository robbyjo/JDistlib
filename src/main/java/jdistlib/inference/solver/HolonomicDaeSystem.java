/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Mechanical index-3 DAE described by acceleration and holonomic constraints. */
public interface HolonomicDaeSystem {
	/** Number of independent position constraints. */
	int constraintCount();
	/** Writes unconstrained acceleration into {@code acceleration}. */
	void acceleration(double time, double[] position, double[] velocity,
			double[] parameters, double[] data, double[] acceleration);
	/** Writes {@code g(time, position) = 0} into {@code residual}. */
	void constraints(double time, double[] position, double[] parameters,
			double[] data, double[] residual);
}
