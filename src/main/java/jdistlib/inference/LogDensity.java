/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** An unnormalized log density on an unconstrained Euclidean state space. */
@FunctionalInterface
public interface LogDensity {
	double logDensity(double[] state);
}
