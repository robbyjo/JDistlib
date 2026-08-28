/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Reversible mapping between parameter/auxiliary pairs of equal total dimension. */
public interface DimensionMatchingTransformation {
	DimensionMatchingResult forward(ReversibleJumpState state, double[] auxiliary);
	DimensionMatchingResult inverse(ReversibleJumpState state, double[] auxiliary);
	double logAbsJacobian(ReversibleJumpState state, double[] auxiliary, boolean forward);
}
