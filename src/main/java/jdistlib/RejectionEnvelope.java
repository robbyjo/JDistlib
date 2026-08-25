/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/**
 * Proposal distribution and certified majorization constant for rejection
 * sampling. Implementations promise {@code target(x) <= M * proposal(x)}.
 */
public interface RejectionEnvelope {
	/** Draws from the normalized proposal distribution. */
	double sample(RandomEngine random);
	/** Returns the normalized proposal log-density at x. */
	double logProposalDensity(double x);
	/** Returns log(M) in the majorization promise. */
	double getLogMajorizationConstant();
}
