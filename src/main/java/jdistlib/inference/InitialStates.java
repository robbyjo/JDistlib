/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Deterministic validation, constrained initialization, and bounded retry helpers. */
public final class InitialStates {
	private InitialStates() {}
	public static double[] named(BayesianModel model, Map<String, double[]> constrained) {
		return validate(model, model.unconstrain(constrained));
	}
	public static double[] retry(LogDensity target, double[] center, double radius,
			int maximumAttempts, RandomEngine random) {
		if (target == null || center == null || !(radius >= 0.0)
				|| maximumAttempts < 1 || random == null)
			throw new IllegalArgumentException("invalid initialization arguments");
		if (Double.isFinite(target.logDensity(center))) return center.clone();
		for (int attempt = 0; attempt < maximumAttempts; attempt++) {
			double[] candidate = center.clone();
			for (int i = 0; i < candidate.length; i++)
				candidate[i] += radius * (2.0 * random.nextDouble() - 1.0);
			if (Double.isFinite(target.logDensity(candidate))) return candidate;
		}
		throw new IllegalArgumentException("no finite initial state after "
				+ maximumAttempts + " deterministic seeded attempts");
	}
	private static double[] validate(LogDensity target, double[] state) {
		if (!Double.isFinite(target.logDensity(state)))
			throw new IllegalArgumentException("named initial state has non-finite density");
		return state;
	}
}
