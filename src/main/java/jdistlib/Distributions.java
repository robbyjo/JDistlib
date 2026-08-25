/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;

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
}
