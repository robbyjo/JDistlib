/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Reproducible aggregation, product/ratio, compound-sum, and scenario helpers. */
public final class DistributionAggregation {
	private DistributionAggregation() {}

	public static DistributionApproximation convolution(GenericDistribution first,
			GenericDistribution second, int draws, long seed) {
		return weightedSum(new GenericDistribution[] {first, second}, new double[] {1.0, 1.0}, draws, seed);
	}

	public static DistributionApproximation weightedSum(GenericDistribution[] distributions,
			double[] weights, int draws, long seed) {
		validate(distributions, weights, draws);
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		for (int i = 0; i < draws; i++) {
			double value = 0.0;
			for (int j = 0; j < distributions.length; j++)
				value += weights[j] * inverseSample(distributions[j], random);
			sample[i] = value;
		}
		return approximation(sample, seed, "reproducible-Monte-Carlo-weighted-sum");
	}

	public static DistributionApproximation compoundSum(GenericDistribution count,
			GenericDistribution severity, int draws, int maximumCount, long seed) {
		if (count == null || severity == null || draws < 100 || maximumCount < 1)
			throw new IllegalArgumentException("laws are required, draws >= 100, maximumCount >= 1");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		int truncated = 0;
		for (int i = 0; i < draws; i++) {
			int events = (int) Math.max(0.0, Math.rint(inverseSample(count, random)));
			if (events > maximumCount) { events = maximumCount; truncated++; }
			double total = 0.0;
			for (int event = 0; event < events; event++) total += inverseSample(severity, random);
			sample[i] = total;
		}
		double error = Math.max(1.0 / Math.sqrt(draws), truncated / (double) draws);
		return new DistributionApproximation(new EmpiricalDistribution(sample),
				new NumericalEstimate(1.0, error, truncated == 0, draws,
						"reproducible-Monte-Carlo-compound-sum",
						truncated == 0 ? "" : truncated + " counts truncated at maximumCount"), seed);
	}

	public static DistributionApproximation product(GenericDistribution first,
			GenericDistribution second, int draws, long seed) {
		return binary(first, second, draws, seed, false);
	}

	public static DistributionApproximation ratio(GenericDistribution numerator,
			GenericDistribution denominator, int draws, long seed) {
		return binary(numerator, denominator, draws, seed, true);
	}

	/** Applies caller-supplied scenarios to a base law with explicit seed provenance. */
	public static DistributionApproximation scenario(GenericDistribution base,
			ScenarioTransformation transformation, int draws, long seed) {
		if (base == null || transformation == null || draws < 100)
			throw new IllegalArgumentException("base/transformation required and draws >= 100");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		for (int i = 0; i < draws; i++) sample[i] = transformation.apply(inverseSample(base, random));
		return approximation(sample, seed, "reproducible-Monte-Carlo-scenario");
	}

	public interface ScenarioTransformation { double apply(double value); }

	private static DistributionApproximation binary(GenericDistribution first,
			GenericDistribution second, int draws, long seed, boolean ratio) {
		if (first == null || second == null || draws < 100)
			throw new IllegalArgumentException("laws are required and draws >= 100");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		int rejected = 0;
		for (int i = 0; i < draws; i++) {
			double a = inverseSample(first, random);
			double b = inverseSample(second, random);
			if (ratio && b == 0.0) { i--; rejected++; if (rejected > draws) throw new IllegalArgumentException("denominator has excessive mass at zero"); }
			else sample[i] = ratio ? a / b : a * b;
		}
		return approximation(sample, seed, ratio ? "reproducible-Monte-Carlo-ratio" : "reproducible-Monte-Carlo-product");
	}

	private static DistributionApproximation approximation(double[] sample, long seed, String strategy) {
		return new DistributionApproximation(new EmpiricalDistribution(sample),
				new NumericalEstimate(1.0, 1.0 / Math.sqrt(sample.length), true,
						sample.length, strategy, "empirical Monte Carlo approximation"), seed);
	}

	private static double inverseSample(GenericDistribution distribution, RandomEngine random) {
		double probability;
		do probability = random.nextDouble(); while (probability == 0.0);
		return distribution.quantile(probability, true, false);
	}

	private static void validate(GenericDistribution[] distributions, double[] weights, int draws) {
		if (distributions == null || weights == null || distributions.length == 0
				|| distributions.length != weights.length || draws < 100)
			throw new IllegalArgumentException("matching nonempty laws/weights and draws >= 100 are required");
		for (int i = 0; i < weights.length; i++) if (distributions[i] == null || !Double.isFinite(weights[i]))
			throw new IllegalArgumentException("laws and finite weights are required");
	}
}
