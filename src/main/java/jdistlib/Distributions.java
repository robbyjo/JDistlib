/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.UnivariateFunction;

/** Concise factories for composing scalar distribution objects. */
public final class Distributions {
	private Distributions() {}
	public static MixtureDistribution mixture(double[] weights,
			GenericDistribution... components) {
		return new MixtureDistribution(weights, components);
	}
	public static TruncatedContinuousDistribution truncate(
			GenericDistribution base, double lower, double upper) {
		return new TruncatedContinuousDistribution(base, lower, upper);
	}
	public static CensoredDistribution censor(GenericDistribution base,
			double lower, double upper) {
		return new CensoredDistribution(base, lower, upper);
	}
	public static MonotoneTransformDistribution affine(GenericDistribution base,
			double shift, double scale) {
		return MonotoneTransformDistribution.affine(base, shift, scale);
	}
	/**
	 * Creates the distribution induced by a differentiable strictly monotone
	 * transformation of an existing distribution.
	 *
	 * @param base distribution of X
	 * @param forward h(x), producing Y
	 * @param inverse inverse transform h^-1(y)
	 * @param logAbsInverseDerivative log |d h^-1(y) / dy|
	 * @param increasing whether h is increasing
	 * @param lower lower bound of Y
	 * @param upper upper bound of Y
	 */
	public static MonotoneTransformDistribution transform(
			GenericDistribution base, UnivariateFunction forward,
			UnivariateFunction inverse,
			UnivariateFunction logAbsInverseDerivative,
			boolean increasing, double lower, double upper) {
		return new MonotoneTransformDistribution(base, forward, inverse,
				logAbsInverseDerivative, increasing, lower, upper);
	}
}
