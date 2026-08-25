/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** A uniform proposal for a finite interval with a certified density bound. */
public final class UniformRejectionEnvelope implements RejectionEnvelope {
	private final double lower;
	private final double upper;
	private final double logWidth;
	private final double logDensityUpperBound;

	/**
	 * @param lower finite proposal lower bound
	 * @param upper finite proposal upper bound
	 * @param logDensityUpperBound certified upper bound on the normalized target
	 *        log-density throughout the interval
	 */
	public UniformRejectionEnvelope(double lower, double upper,
			double logDensityUpperBound) {
		if (!Double.isFinite(lower) || !Double.isFinite(upper) || !(lower < upper)) {
			throw new IllegalArgumentException("uniform envelope bounds must be finite and ordered");
		}
		if (!Double.isFinite(logDensityUpperBound)) {
			throw new IllegalArgumentException("logDensityUpperBound must be finite");
		}
		double width = upper - lower;
		if (!Double.isFinite(width)) {
			throw new IllegalArgumentException(
					"uniform envelope width must be representable");
		}
		this.lower = lower;
		this.upper = upper;
		this.logWidth = Math.log(width);
		this.logDensityUpperBound = logDensityUpperBound;
	}

	@Override public double sample(RandomEngine random) {
		double unit = random.nextDouble();
		return lower * (1.0 - unit) + upper * unit;
	}

	@Override public double logProposalDensity(double x) {
		return x >= lower && x <= upper ? -logWidth : Double.NEGATIVE_INFINITY;
	}

	@Override public double getLogMajorizationConstant() {
		return logDensityUpperBound + logWidth;
	}

	public double getLowerBound() { return lower; }
	public double getUpperBound() { return upper; }
	public double getLogDensityUpperBound() { return logDensityUpperBound; }
}
