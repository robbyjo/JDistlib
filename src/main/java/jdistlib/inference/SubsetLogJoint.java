/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Complete normalized log joint for one active-variable subset. */
@FunctionalInterface
public interface SubsetLogJoint {
	double logJoint(double[] commonParameters, int[] activeCandidates, double[] activeCoefficients);
}
