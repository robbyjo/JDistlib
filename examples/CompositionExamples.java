/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package examples;

import jdistlib.CensoredDistribution;
import jdistlib.Distributions;
import jdistlib.MixtureDistribution;
import jdistlib.MonotoneTransformDistribution;
import jdistlib.Normal;
import jdistlib.TruncatedContinuousDistribution;

/** Compilable examples for the composition and transformation tutorial. */
public final class CompositionExamples {
	private CompositionExamples() {}

	/** A log-normal law built explicitly as exp(X), X normal. */
	public static MonotoneTransformDistribution exponentialTransform() {
		return Distributions.transform(
				new Normal(0.0, 0.5),
				Math::exp,
				Math::log,
				y -> -Math.log(y),
				true,
				0.0,
				Double.POSITIVE_INFINITY);
	}

	/** Builds a mixture -> truncation -> affine -> censoring design. */
	public static CensoredDistribution measurementDesign() {
		MixtureDistribution latent = Distributions.mixture(
				new double[] {0.8, 0.2},
				new Normal(50.0, 8.0),
				new Normal(75.0, 12.0));
		TruncatedContinuousDistribution physicallyPossible =
				Distributions.truncate(latent, 0.0,
						Double.POSITIVE_INFINITY);
		MonotoneTransformDistribution calibrated =
				Distributions.affine(physicallyPossible, 2.0, 1.1);
		return Distributions.censor(calibrated, 5.0, 100.0);
	}

	public static void main(String[] arguments) {
		MonotoneTransformDistribution positive = exponentialTransform();
		System.out.println("P(Y <= 1) = " + positive.cumulative(1.0));
		System.out.println("90th percentile = " + positive.quantile(0.9));

		CensoredDistribution observed = measurementDesign();
		System.out.println("P(observed at upper limit) = "
				+ observed.getUpperAtomProbability());
		System.out.println("P(observed <= 60) = "
				+ observed.cumulative(60.0));
	}
}
