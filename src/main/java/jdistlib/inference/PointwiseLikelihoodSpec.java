/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Package-private compiled pointwise likelihood component. */
final class PointwiseLikelihoodSpec {
	final ObservationMetadata metadata;
	final PointwiseLogLikelihoodEvaluator evaluator;
	PointwiseLikelihoodSpec(ObservationMetadata metadata, PointwiseLogLikelihoodEvaluator evaluator) {
		this.metadata = metadata; this.evaluator = evaluator;
	}
}
