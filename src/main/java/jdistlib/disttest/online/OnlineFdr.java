/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest.online;

/** Utilities shared by online false-discovery-rate controllers. */
public final class OnlineFdr {
	private OnlineFdr() {}

	/**
	 * Returns a finite, normalized, nonincreasing gamma sequence proportional
	 * to {@code index^-exponent}; it is padded with zeros beyond the horizon.
	 */
	public static double[] polynomialGamma(int horizon, double exponent) {
		if (horizon <= 0)
			throw new IllegalArgumentException("horizon must be positive");
		if (!Double.isFinite(exponent) || exponent <= 1.0)
			throw new IllegalArgumentException("exponent must exceed one");
		double[] gamma = new double[horizon];
		double sum = 0.0;
		for (int i = 0; i < horizon; i++) {
			gamma[i] = Math.pow(i + 1.0, -exponent);
			sum += gamma[i];
		}
		for (int i = 0; i < horizon; i++) gamma[i] /= sum;
		return gamma;
	}

	static double[] validateGamma(double[] gamma) {
		if (gamma == null || gamma.length == 0)
			throw new IllegalArgumentException("gamma must not be empty");
		double[] copy = gamma.clone();
		double sum = 0.0;
		double previous = Double.POSITIVE_INFINITY;
		for (double value : copy) {
			if (!Double.isFinite(value) || value < 0.0)
				throw new IllegalArgumentException(
						"gamma values must be finite and nonnegative");
			if (value > previous)
				throw new IllegalArgumentException(
						"gamma must be nonincreasing");
			previous = value;
			sum += value;
		}
		if (copy[0] <= 0.0 || sum > 1.0 + 1e-12)
			throw new IllegalArgumentException(
					"gamma must start positive and sum to at most one");
		return copy;
	}

	static double gamma(double[] sequence, long oneBasedIndex) {
		return oneBasedIndex <= 0 || oneBasedIndex > sequence.length ? 0.0
				: sequence[(int) oneBasedIndex - 1];
	}

	static void validateProbability(double value, String name) {
		if (!Double.isFinite(value) || value < 0.0 || value > 1.0)
			throw new IllegalArgumentException(name + " must be in [0, 1]");
	}
}
