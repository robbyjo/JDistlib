/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** L-BFGS path plus local Gaussian draw for robust chain initialization. */
public final class PathfinderInitializer {
	private PathfinderInitializer() {}
	public static PathfinderResult initialize(DifferentiableLogDensity target,
			double[] initial, RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random is required");
		OptimizationResult optimization = LbfgsOptimizer.maximize(target, initial,
				1000, 8, 1e-6);
		double[] mode = optimization.point();
		double[] scale = diagonalScale(target, mode);
		double[] draw = mode.clone();
		for (int i = 0; i < draw.length; i++) draw[i] += scale[i] * random.nextGaussian();
		if (!Double.isFinite(target.logDensity(draw))) draw = mode.clone();
		return new PathfinderResult(draw, mode, scale, optimization);
	}
	private static double[] diagonalScale(DifferentiableLogDensity target, double[] mode) {
		double[] result = new double[mode.length], plusGradient = new double[mode.length];
		double[] minusGradient = new double[mode.length];
		for (int i = 0; i < mode.length; i++) {
			double h = Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(mode[i]));
			double[] plus = mode.clone(), minus = mode.clone(); plus[i] += h; minus[i] -= h;
			target.logDensityAndGradient(plus, plusGradient);
			target.logDensityAndGradient(minus, minusGradient);
			double negativeCurvature = -(plusGradient[i] - minusGradient[i]) / (2.0 * h);
			result[i] = 1.0 / Math.sqrt(Math.max(1e-6, negativeCurvature));
			result[i] = Math.min(100.0, result[i]);
		}
		return result;
	}
}
