/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Supplies the row score vector used by a locally informed sparse proposal. */
@FunctionalInterface
public interface SparseResidualProvider {
	double[] scoreVector(SparseSubsetState state);
}
