/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.Distributions;
import jdistlib.Exponential;
import jdistlib.MixtureDistribution;
import jdistlib.Normal;
import jdistlib.rng.MersenneTwister;

/** Mixture evaluation, quantiles, and reproducible simulation. */
public final class MixtureDistributionExamples {
	private MixtureDistributionExamples() {}
	public static MixtureDistribution responseTimes() {
		return Distributions.mixture(new double[] {0.85, 0.15},
				new Normal(4, 0.8), new Normal(12, 2.5));
	}
	public static MixtureDistribution lifetimeWithEarlyFailures() {
		return Distributions.mixture(new double[] {0.2, 0.8},
				new Exponential(0.5), new Exponential(8));
	}
	public static void main(String[] arguments) {
		MixtureDistribution mixture = responseTimes();
		System.out.println("density=" + mixture.density(6, false));
		System.out.println("within target=" + mixture.cumulative(8));
		System.out.println("p95=" + mixture.quantile(0.95));
		mixture.setRandomEngine(new MersenneTwister(20260827L));
		System.out.println("draws=" + Arrays.toString(mixture.random(5)));
	}
}
