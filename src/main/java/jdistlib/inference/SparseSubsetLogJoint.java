/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Complete normalized log joint for an arbitrary sparse candidate universe. */
@FunctionalInterface
public interface SparseSubsetLogJoint {
	double logJoint(double[] commonParameters, int[] activeCandidates, double[] coefficients);
}
