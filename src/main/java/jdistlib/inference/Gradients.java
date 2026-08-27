/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Finite-difference adapters and gradient validation utilities. */
public final class Gradients {
	private static final double DEFAULT_STEP = Math.cbrt(Math.ulp(1.0));

	private Gradients() {}

	/** Creates a central finite-difference adapter for diagnostic or fallback use. */
	public static DifferentiableLogDensity finiteDifference(LogDensity target) {
		return finiteDifference(target, DEFAULT_STEP);
	}

	public static DifferentiableLogDensity finiteDifference(final LogDensity target,
			final double relativeStep) {
		if (target == null || !(relativeStep > 0.0)
				|| !Double.isFinite(relativeStep)) {
			throw new IllegalArgumentException("target and a finite positive step are required");
		}
		return new DifferentiableLogDensity() {
			@Override public double logDensityAndGradient(double[] state,
					double[] gradient) {
				validate(state, gradient);
				double value = target.logDensity(state);
				double[] work = state.clone();
				for (int i = 0; i < state.length; i++) {
					double step = relativeStep * Math.max(1.0, Math.abs(state[i]));
					work[i] = state[i] + step;
					double right = target.logDensity(work);
					work[i] = state[i] - step;
					double left = target.logDensity(work);
					work[i] = state[i];
					gradient[i] = (right - left) / (2.0 * step);
				}
				return value;
			}
		};
	}

	public static GradientCheckResult check(DifferentiableLogDensity target,
			double[] state, double absoluteTolerance, double relativeTolerance) {
		if (target == null || state == null || !(absoluteTolerance >= 0.0)
				|| !(relativeTolerance >= 0.0)) {
			throw new IllegalArgumentException("invalid gradient-check arguments");
		}
		double[] supplied = new double[state.length];
		double[] reference = new double[state.length];
		double value = target.logDensityAndGradient(state.clone(), supplied);
		finiteDifference((LogDensity) target).logDensityAndGradient(state.clone(), reference);
		double maxAbsolute = 0.0;
		double maxRelative = 0.0;
		int worst = -1;
		boolean passed = Double.isFinite(value);
		for (int i = 0; i < state.length; i++) {
			double absolute = Math.abs(supplied[i] - reference[i]);
			double scale = Math.max(1.0, Math.max(Math.abs(supplied[i]),
					Math.abs(reference[i])));
			double relative = absolute / scale;
			if (!Double.isFinite(absolute) || absolute > maxAbsolute) {
				maxAbsolute = absolute;
				worst = i;
			}
			maxRelative = Math.max(maxRelative, relative);
			if (!Double.isFinite(absolute)
					|| (absolute > absoluteTolerance && relative > relativeTolerance)) {
				passed = false;
			}
		}
		String message = passed ? "gradient agrees with central finite differences"
				: "gradient mismatch at coordinate " + worst + "; supplied="
				+ (worst < 0 ? "n/a" : supplied[worst]) + ", reference="
				+ (worst < 0 ? "n/a" : reference[worst]);
		return new GradientCheckResult(passed, maxAbsolute, maxRelative, worst, message);
	}

	static void validate(double[] state, double[] gradient) {
		if (state == null || gradient == null || state.length != gradient.length) {
			throw new IllegalArgumentException("state and gradient dimensions must match");
		}
		Arrays.fill(gradient, 0.0);
	}
}
