/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Model contract required by pointwise predictive assessment. */
public interface PointwiseLogLikelihood {
	ObservationMetadata observationMetadata();
	double[] pointwiseLogLikelihood(double[] unconstrainedState);

	default PointwiseLogLikelihoodDraws extractPointwiseLogLikelihood(ChainResult... chains) {
		return PointwiseLogLikelihoodDraws.extract(this, chains);
	}
}
