/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Residual function {@code F(state, parameters, data) = 0}. */
@FunctionalInterface
public interface AlgebraicSystem {
	/** Writes the residual into {@code residual}. */
	void evaluate(double[] state, double[] parameters, double[] data, double[] residual);
}
