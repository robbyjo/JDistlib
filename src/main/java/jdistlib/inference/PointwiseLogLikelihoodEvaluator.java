/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Evaluates ordered observation-level log-likelihood contributions. */
@FunctionalInterface
public interface PointwiseLogLikelihoodEvaluator {
	double[] evaluate(ModelState state);
}
