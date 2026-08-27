/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** First-order ordinary differential equation {@code y' = f(t,y)}. */
@FunctionalInterface
public interface OdeSystem {
	/** Writes the state derivative into {@code derivative}. */
	void derivatives(double time, double[] state, double[] parameters,
			double[] data, double[] derivative);
}
