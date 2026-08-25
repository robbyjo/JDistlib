/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.math.UnivariateFunction;
import jdistlib.rng.RandomEngine;

/**
 * Adaptive tangent-envelope rejection sampler for a caller-certified
 * differentiable log-concave density on finite support.
 */
public final class AdaptiveRejectionSampler {
	private final NumericalContinuousDistribution distribution;
	private final UnivariateFunction logDerivative;
	private final double lower;
	private final double upper;
	private final int maximumKnots;
	private final int maximumAttempts;
	private double[] knots;
	private double[] values;
	private double[] slopes;
	private double[] intersections;
	private double[] logMasses;

	public AdaptiveRejectionSampler(NumericalContinuousDistribution distribution,
			UnivariateFunction logDerivative, int maximumKnots, int maximumAttempts,
			double... initialPoints) {
		if (distribution == null || logDerivative == null) {
			throw new IllegalArgumentException("distribution and log derivative are required");
		}
		lower = distribution.getLowerBound();
		upper = distribution.getUpperBound();
		if (!Double.isFinite(lower) || !Double.isFinite(upper)) {
			throw new IllegalArgumentException(
					"adaptive rejection sampling currently requires finite support");
		}
		if (maximumKnots < 3 || maximumKnots > 10000 || maximumAttempts < 1) {
			throw new IllegalArgumentException("invalid adaptive rejection limits");
		}
		if (initialPoints == null || initialPoints.length < 2
				|| initialPoints.length > maximumKnots) {
			throw new IllegalArgumentException("at least two initial interior points are required");
		}
		this.distribution = distribution;
		this.logDerivative = logDerivative;
		this.maximumKnots = maximumKnots;
		this.maximumAttempts = maximumAttempts;
		knots = initialPoints.clone();
		Arrays.sort(knots);
		for (int i = 0; i < knots.length; i++) {
			if (!(knots[i] > lower && knots[i] < upper)
					|| (i > 0 && knots[i] == knots[i - 1])) {
				throw new IllegalArgumentException(
						"adaptive rejection points must be unique and interior");
			}
		}
		rebuild();
	}

	public synchronized double sample(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random must not be null");
		for (int attempt = 0; attempt < maximumAttempts; attempt++) {
			int segment = chooseSegment(random.nextDouble());
			double candidate = sampleSegment(segment, random.nextDouble());
			candidate = Math.max(Math.nextUp(lower),
					Math.min(Math.nextDown(upper), candidate));
			double upperLogDensity = slopes[segment] * candidate
					+ values[segment] - slopes[segment] * knots[segment];
			double target = distribution.density(candidate, true);
			if (!Double.isFinite(target)) {
				throw new IllegalStateException("log density was non-finite at sampled x="
						+ candidate);
			}
			if (target > upperLogDensity
					+ 1e-10 * Math.max(1.0, Math.abs(target))) {
				throw new IllegalStateException(
						"log-concavity/tangent-envelope promise failed at x=" + candidate);
			}
			if (Math.log(random.nextDouble()) <= target - upperLogDensity) {
				return candidate;
			}
			if (knots.length < maximumKnots) insert(candidate);
		}
		throw new IllegalStateException(
				"adaptive rejection attempt budget was exhausted");
	}

	public synchronized int getKnotCount() { return knots.length; }
	public int getMaximumKnots() { return maximumKnots; }
	public int getMaximumAttempts() { return maximumAttempts; }

	private void insert(double point) {
		int location = Arrays.binarySearch(knots, point);
		if (location >= 0) return;
		location = -location - 1;
		double[] expanded = new double[knots.length + 1];
		System.arraycopy(knots, 0, expanded, 0, location);
		expanded[location] = point;
		System.arraycopy(knots, location, expanded, location + 1,
				knots.length - location);
		knots = expanded;
		rebuild();
	}

	private void rebuild() {
		int n = knots.length;
		values = new double[n];
		slopes = new double[n];
		for (int i = 0; i < n; i++) {
			values[i] = distribution.density(knots[i], true);
			slopes[i] = logDerivative.eval(knots[i]);
			if (!Double.isFinite(values[i]) || !Double.isFinite(slopes[i])) {
				throw new IllegalArgumentException(
						"log density and derivative must be finite at every knot");
			}
			if (i > 0 && slopes[i] >= slopes[i - 1]) {
				throw new IllegalArgumentException(
						"log derivative must decrease strictly across knots");
			}
		}
		intersections = new double[n + 1];
		intersections[0] = lower;
		intersections[n] = upper;
		for (int i = 1; i < n; i++) {
			double leftIntercept = values[i - 1] - slopes[i - 1] * knots[i - 1];
			double rightIntercept = values[i] - slopes[i] * knots[i];
			intersections[i] = (rightIntercept - leftIntercept)
					/ (slopes[i - 1] - slopes[i]);
			if (!(intersections[i] > intersections[i - 1]
					&& intersections[i] < upper)) {
				throw new IllegalArgumentException("invalid log-concave tangent intersections");
			}
		}
		logMasses = new double[n];
		for (int i = 0; i < n; i++) {
			double intercept = values[i] - slopes[i] * knots[i];
			logMasses[i] = logIntegral(slopes[i], intercept,
					intersections[i], intersections[i + 1]);
		}
	}

	private int chooseSegment(double uniform) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (double mass : logMasses) maximum = Math.max(maximum, mass);
		double total = 0.0;
		for (double mass : logMasses) total += Math.exp(mass - maximum);
		double target = uniform * total;
		for (int i = 0; i < logMasses.length; i++) {
			target -= Math.exp(logMasses[i] - maximum);
			if (target <= 0.0) return i;
		}
		return logMasses.length - 1;
	}

	private double sampleSegment(int segment, double uniform) {
		double slope = slopes[segment];
		double left = intersections[segment];
		double right = intersections[segment + 1];
		if (Math.abs(slope) < 1e-12) return left + uniform * (right - left);
		if (slope > 0.0) {
			return right + Math.log(uniform
					+ (1.0 - uniform) * Math.exp(slope * (left - right))) / slope;
		}
		return left + Math.log((1.0 - uniform)
				+ uniform * Math.exp(slope * (right - left))) / slope;
	}

	private static double logIntegral(double slope, double intercept,
			double left, double right) {
		if (Math.abs(slope) < 1e-12) {
			return intercept + Math.log(right - left);
		}
		if (slope > 0.0) {
			return intercept + slope * right
					+ Math.log1p(-Math.exp(slope * (left - right)))
					- Math.log(slope);
		}
		return intercept + slope * left
				+ Math.log1p(-Math.exp(slope * (right - left)))
				- Math.log(-slope);
	}
}
