/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Differential-algebraic residual {@code F(t, y, y') = 0}. */
@FunctionalInterface
public interface DaeSystem {
	/** Writes the DAE residual into {@code residual}. */
	void residual(double time, double[] state, double[] stateDerivative,
			double[] parameters, double[] data, double[] residual);
}
