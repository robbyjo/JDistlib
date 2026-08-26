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
import jdistlib.rng.MersenneTwister;

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

	/** A two-regime response-time mixture transformed from log seconds. */
	public static MonotoneTransformDistribution responseTimeMixture() {
		MixtureDistribution logSeconds = Distributions.mixture(
				new double[] {0.85, 0.15},
				new Normal(Math.log(4.0), 0.30),
				new Normal(Math.log(20.0), 0.45));
		return Distributions.transform(
				logSeconds,
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

		MonotoneTransformDistribution responseTime = responseTimeMixture();
		double densityAtTen = responseTime.density(10.0, false);
		double withinTen = responseTime.cumulative(10.0);
		double percentile95 = responseTime.quantile(0.95);
		responseTime.setRandomEngine(new MersenneTwister(20260826L));
		double[] scenarios = responseTime.random(1_000);
		System.out.println("density at 10 seconds = " + densityAtTen);
		System.out.println("P(response <= 10 seconds) = " + withinTen);
		System.out.println("95th percentile = " + percentile95);
		System.out.println("simulated scenarios = " + scenarios.length);

		CensoredDistribution observed = measurementDesign();
		System.out.println("P(observed at upper limit) = "
				+ observed.getUpperAtomProbability());
		System.out.println("P(observed <= 60) = "
				+ observed.cumulative(60.0));
	}
}
