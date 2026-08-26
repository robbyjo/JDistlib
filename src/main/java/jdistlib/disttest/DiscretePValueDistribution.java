/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest;

import java.util.Arrays;

/**
 * Finite null distribution of a discrete p-value.
 *
 * <p>The CDF must be super-uniform ({@code F(t) <= t}) so it is a valid
 * p-value distribution. For a conventional exact p-value whose CDF equals
 * each attainable p-value at support points, use {@link #exact(double[])}.</p>
 */
public final class DiscretePValueDistribution {
	private final double[] support;
	private final double[] cdf;

	public DiscretePValueDistribution(double[] support, double[] cdf) {
		if (support == null || cdf == null || support.length == 0
				|| support.length != cdf.length)
			throw new IllegalArgumentException(
					"support and CDF must be nonempty and have equal length");
		this.support = support.clone();
		this.cdf = cdf.clone();
		double previousSupport = -1.0;
		double previousCdf = 0.0;
		for (int i = 0; i < support.length; i++) {
			double point = this.support[i];
			double probability = this.cdf[i];
			if (!Double.isFinite(point) || point < 0.0 || point > 1.0
					|| point <= previousSupport)
				throw new IllegalArgumentException(
						"support must be finite, strictly increasing, and in [0, 1]");
			if (!Double.isFinite(probability) || probability < previousCdf
					|| probability > point + 1e-12)
				throw new IllegalArgumentException(
						"CDF must be nondecreasing and super-uniform");
			previousSupport = point;
			previousCdf = probability;
		}
		int last = support.length - 1;
		if (this.support[last] != 1.0 || this.cdf[last] != 1.0)
			throw new IllegalArgumentException(
					"support and CDF must both end at one");
	}

	/** Builds the common exact-p-value distribution with F(t)=t on its support. */
	public static DiscretePValueDistribution exact(double[] support) {
		return new DiscretePValueDistribution(support, support);
	}

	/** Returns a defensive copy of the attainable p-values. */
	public double[] getSupport() { return support.clone(); }

	/** Evaluates the right-continuous CDF. */
	public double cdf(double value) {
		if (Double.isNaN(value)) return Double.NaN;
		int index = Arrays.binarySearch(support, value);
		if (index >= 0) return cdf[index];
		index = -index - 2;
		return index < 0 ? 0.0 : cdf[index];
	}
}
