/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Named NIG specialization of the generalized-hyperbolic family. */
public final class NormalInverseGaussianDistribution extends GeneralizedHyperbolicDistribution {
	public NormalInverseGaussianDistribution(double alpha, double beta, double delta, double mu) {
		super(-0.5, alpha, beta, delta, mu);
	}
}
