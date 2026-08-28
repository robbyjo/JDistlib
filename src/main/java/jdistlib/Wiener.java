/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/**
 * Four-parameter Wiener first-passage (drift-diffusion) density used by Stan.
 * The density is the upper-response component and therefore integrates to the
 * corresponding boundary-choice probability rather than to one.
 */
public final class Wiener {
	private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
	private static final double SERIES_LOG_CUTOFF = 48.0;

	private Wiener() {}

	public static double density(double y, double boundary, double nondecision,
			double bias, double drift, boolean giveLog) {
		double logDensity = logDensity(y, boundary, nondecision, bias, drift);
		return giveLog ? logDensity : Math.exp(logDensity);
	}

	private static double logDensity(double y, double boundary,
			double nondecision, double bias, double drift) {
		if (Double.isNaN(y) || Double.isNaN(boundary) ||
				Double.isNaN(nondecision) || Double.isNaN(bias) ||
				Double.isNaN(drift) || !(boundary > 0.0) ||
				!Double.isFinite(boundary) || !(nondecision >= 0.0) ||
				!Double.isFinite(nondecision) || bias < 0.0 || bias > 1.0 ||
				!Double.isFinite(drift)) return Double.NaN;
		if (!(y > nondecision) || bias == 0.0 || bias == 1.0)
			return Double.NEGATIVE_INFINITY;
		double decisionTime = y - nondecision;
		double scaledTime = decisionTime / (boundary * boundary);
		double standardLog = scaledTime < 0.25
				? smallTimeLogDensity(scaledTime, bias)
				: largeTimeLogDensity(scaledTime, bias);
		if (!Double.isFinite(standardLog)) {
			// The alternate expansion is slower here but is a useful cancellation guard.
			standardLog = scaledTime < 0.25
					? largeTimeLogDensity(scaledTime, bias)
					: smallTimeLogDensity(scaledTime, bias);
		}
		return standardLog - 2.0 * Math.log(boundary) -
				drift * boundary * bias - 0.5 * drift * drift * decisionTime;
	}

	private static double smallTimeLogDensity(double time, double bias) {
		double radius = Math.sqrt(bias * bias + 2.0 * time * SERIES_LOG_CUTOFF);
		int terms = Math.max(2, (int) Math.ceil((radius + 1.0) / 2.0) + 2);
		SignedLogSum sum = new SignedLogSum();
		for (int k = -terms; k <= terms; k++) {
			double distance = bias + 2.0 * k;
			if (distance == 0.0) continue;
			sum.add(Math.signum(distance), Math.log(Math.abs(distance)) -
					0.5 * distance * distance / time);
		}
		return sum.logValue() - 0.5 * LOG_TWO_PI - 1.5 * Math.log(time);
	}

	private static double largeTimeLogDensity(double time, double bias) {
		int terms = Math.max(4, (int) Math.ceil(Math.sqrt(
				2.0 * SERIES_LOG_CUTOFF / (Math.PI * Math.PI * time))) + 2);
		SignedLogSum sum = new SignedLogSum();
		for (int k = 1; k <= terms; k++) {
			double sine = Math.sin(k * Math.PI * bias);
			if (sine == 0.0) continue;
			sum.add(Math.signum(sine), Math.log(k * Math.abs(sine)) -
					0.5 * k * k * Math.PI * Math.PI * time);
		}
		return Math.log(Math.PI) + sum.logValue();
	}

	/** Boundary-choice probability, equal to the integral of {@link #density}. */
	public static double boundaryProbability(double boundary, double bias,
			double drift) {
		if (!(boundary > 0.0) || !Double.isFinite(boundary) || bias < 0.0 ||
				bias > 1.0 || !Double.isFinite(bias) || !Double.isFinite(drift))
			return Double.NaN;
		double x = 2.0 * drift * boundary;
		if (Math.abs(x) < 1e-8) return 1.0 - bias;
		double result = x > 0.0
				? (Math.exp(-x * bias) - Math.exp(-x)) / -Math.expm1(-x)
				: Math.expm1(x * (1.0 - bias)) / Math.expm1(x);
		return Math.max(0.0, Math.min(1.0, result));
	}

	/** Defective CDF matching the upper-boundary density component. */
	public static double cumulative(double y, double boundary, double nondecision,
			double bias, double drift) {
		if (!(y > nondecision)) return y == y ? 0.0 : Double.NaN;
		double mass = boundaryProbability(boundary, bias, drift);
		if (!Double.isFinite(mass) || !(nondecision >= 0.0)) return Double.NaN;
		double decisionTime = y - nondecision;
		double characteristic = boundary * boundary *
				Math.max(bias * bias, Math.ulp(1.0));
		double logLower = Math.max(-744.0, Math.log(characteristic) - 28.0);
		double logUpper = Math.log(decisionTime);
		if (!(logLower < logUpper)) return 0.0;
		double value = adaptiveSimpson(logLower, logUpper, boundary, bias, drift,
				1e-11 * Math.max(mass, 1e-6), 18);
		return Math.max(0.0, Math.min(mass, value));
	}

	/**
	 * Draws a first-passage time conditional on the modeled upper response.
	 * Inversion is deliberately numerical: it uses the same stable two-series
	 * density as likelihood evaluation, avoiding time-discretization bias.
	 */
	public static double random(double boundary, double nondecision, double bias,
			double drift, RandomEngine random) {
		if (random == null) return Double.NaN;
		double mass = boundaryProbability(boundary, bias, drift);
		if (!(mass > 0.0) || !(nondecision >= 0.0) ||
				!Double.isFinite(nondecision)) return Double.NaN;
		if (bias == 0.0) return nondecision;
		double target = Math.max(Math.nextUp(0.0), Math.min(
				Math.nextAfter(mass, 0.0), random.nextDouble() * mass));
		double scale = boundary * boundary;
		double lower = Math.log(scale) + 2.0 * Math.log(Math.max(bias, 1e-12)) - 12.0;
		double upper = Math.log(scale) + 2.0;
		while (cumulative(nondecision + Math.exp(upper), boundary, nondecision,
				bias, drift) < target && upper < Math.log(Double.MAX_VALUE) - 2.0)
			upper += 2.0;
		for (int iteration = 0; iteration < 64; iteration++) {
			double middle = 0.5 * (lower + upper);
			double probability = cumulative(nondecision + Math.exp(middle), boundary,
					nondecision, bias, drift);
			if (probability < target) lower = middle; else upper = middle;
		}
		return nondecision + Math.exp(0.5 * (lower + upper));
	}

	private static double adaptiveSimpson(double lower, double upper,
			double boundary, double bias, double drift, double tolerance, int depth) {
		double middle = 0.5 * (lower + upper);
		double fl = logTimeIntegrand(lower, boundary, bias, drift);
		double fm = logTimeIntegrand(middle, boundary, bias, drift);
		double fu = logTimeIntegrand(upper, boundary, bias, drift);
		double whole = (upper - lower) * (fl + 4.0 * fm + fu) / 6.0;
		return adaptiveSimpsonRecursive(lower, upper, fl, fm, fu, whole, boundary,
				bias, drift, tolerance, depth);
	}

	private static double adaptiveSimpsonRecursive(double lower, double upper,
			double fl, double fm, double fu, double whole, double boundary,
			double bias, double drift, double tolerance, int depth) {
		double middle = 0.5 * (lower + upper);
		double leftMiddle = 0.5 * (lower + middle);
		double rightMiddle = 0.5 * (middle + upper);
		double fLeft = logTimeIntegrand(leftMiddle, boundary, bias, drift);
		double fRight = logTimeIntegrand(rightMiddle, boundary, bias, drift);
		double left = (middle - lower) * (fl + 4.0 * fLeft + fm) / 6.0;
		double right = (upper - middle) * (fm + 4.0 * fRight + fu) / 6.0;
		double difference = left + right - whole;
		if (depth == 0 || Math.abs(difference) <= 15.0 * tolerance)
			return left + right + difference / 15.0;
		return adaptiveSimpsonRecursive(lower, middle, fl, fLeft, fm, left,
				boundary, bias, drift, tolerance / 2.0, depth - 1) +
				adaptiveSimpsonRecursive(middle, upper, fm, fRight, fu, right,
						boundary, bias, drift, tolerance / 2.0, depth - 1);
	}

	private static double logTimeIntegrand(double logTime, double boundary,
			double bias, double drift) {
		double time = Math.exp(logTime);
		return Math.exp(logDensity(time, boundary, 0.0, bias, drift) + logTime);
	}

	private static final class SignedLogSum {
		double positive = Double.NEGATIVE_INFINITY;
		double negative = Double.NEGATIVE_INFINITY;
		void add(double sign, double logMagnitude) {
			if (sign > 0.0) positive = logAdd(positive, logMagnitude);
			else negative = logAdd(negative, logMagnitude);
		}
		double logValue() {
			if (!(positive > negative)) return Double.NEGATIVE_INFINITY;
			if (negative == Double.NEGATIVE_INFINITY) return positive;
			return positive + Math.log1p(-Math.exp(negative - positive));
		}
		private static double logAdd(double left, double right) {
			if (left == Double.NEGATIVE_INFINITY) return right;
			if (right == Double.NEGATIVE_INFINITY) return left;
			double maximum = Math.max(left, right);
			return maximum + Math.log1p(Math.exp(Math.min(left, right) - maximum));
		}
	}
}
