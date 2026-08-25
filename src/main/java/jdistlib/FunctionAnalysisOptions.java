/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.math.IntegrationOptions;

/** Immutable settings for probability-kernel sanity analysis. */
public final class FunctionAnalysisOptions {
	private final int sampleCount;
	private final int repeatabilityChecks;
	private final double discontinuityRatio;
	private final double dynamicRangeOrders;
	private final int randomizedProbeBudget;
	private final int adaptiveProbeRounds;
	private final long randomSeed;
	private final ConstructionPolicy constructionPolicy;
	private final IntegrationOptions integrationOptions;

	private FunctionAnalysisOptions(Builder builder) {
		sampleCount = builder.sampleCount;
		repeatabilityChecks = builder.repeatabilityChecks;
		discontinuityRatio = builder.discontinuityRatio;
		dynamicRangeOrders = builder.dynamicRangeOrders;
		randomizedProbeBudget = builder.randomizedProbeBudget;
		adaptiveProbeRounds = builder.adaptiveProbeRounds;
		randomSeed = builder.randomSeed;
		constructionPolicy = builder.constructionPolicy;
		integrationOptions = builder.integrationOptions;
	}

	public static Builder builder() { return new Builder(); }
	public static FunctionAnalysisOptions defaults() { return builder().build(); }
	public int getSampleCount() { return sampleCount; }
	public int getRepeatabilityChecks() { return repeatabilityChecks; }
	public double getDiscontinuityRatio() { return discontinuityRatio; }
	public double getDynamicRangeOrders() { return dynamicRangeOrders; }
	/** Maximum number of seeded randomized probes performed after the grid. */
	public int getRandomizedProbeBudget() { return randomizedProbeBudget; }
	/** Number of rounds used to focus randomized probes around observed features. */
	public int getAdaptiveProbeRounds() { return adaptiveProbeRounds; }
	/** Seed that makes randomized probing reproducible. */
	public long getRandomSeed() { return randomSeed; }
	public ConstructionPolicy getConstructionPolicy() { return constructionPolicy; }
	public IntegrationOptions getIntegrationOptions() { return integrationOptions; }

	public static final class Builder {
		private int sampleCount = 257;
		private int repeatabilityChecks = 5;
		private double discontinuityRatio = 1e6;
		private double dynamicRangeOrders = 12.0;
		private int randomizedProbeBudget = 128;
		private int adaptiveProbeRounds = 4;
		private long randomSeed = 0x4a446973744c6962L;
		private ConstructionPolicy constructionPolicy = ConstructionPolicy.WARNING;
		private IntegrationOptions integrationOptions = IntegrationOptions.builder()
				.tolerances(1e-9, 1e-9)
				.subdivisions(300)
				.maxEvaluations(250000)
				.method(IntegrationOptions.Method.AUTO)
				.build();

		private Builder() {}

		public Builder sampleCount(int value) { sampleCount = value; return this; }
		public Builder repeatabilityChecks(int value) {
			repeatabilityChecks = value;
			return this;
		}
		public Builder discontinuityRatio(double value) {
			discontinuityRatio = value;
			return this;
		}
		public Builder dynamicRangeOrders(double value) {
			dynamicRangeOrders = value;
			return this;
		}
		/** Sets the total randomized sampling budget; zero disables random probes. */
		public Builder randomizedProbeBudget(int value) {
			randomizedProbeBudget = value;
			return this;
		}

		public Builder adaptiveProbeRounds(int value) {
			adaptiveProbeRounds = value;
			return this;
		}

		public Builder randomSeed(long value) {
			randomSeed = value;
			return this;
		}

		public Builder constructionPolicy(ConstructionPolicy value) {
			constructionPolicy = value;
			return this;
		}
		public Builder integrationOptions(IntegrationOptions value) {
			integrationOptions = value;
			return this;
		}

		public FunctionAnalysisOptions build() {
			if (sampleCount < 9 || sampleCount > 100000) {
				throw new IllegalArgumentException(
						"sampleCount must be between 9 and 100000");
			}
			if (repeatabilityChecks < 0) {
				throw new IllegalArgumentException("repeatabilityChecks must be nonnegative");
			}
			if (!(discontinuityRatio > 1.0)) {
				throw new IllegalArgumentException("discontinuityRatio must exceed one");
			}
			if (!(dynamicRangeOrders > 0.0)) {
				throw new IllegalArgumentException("dynamicRangeOrders must be positive");
			}
			if (randomizedProbeBudget < 0 || randomizedProbeBudget > 100000) {
				throw new IllegalArgumentException(
						"randomizedProbeBudget must be between 0 and 100000");
			}
			if (adaptiveProbeRounds < 1 || adaptiveProbeRounds > 64) {
				throw new IllegalArgumentException(
						"adaptiveProbeRounds must be between 1 and 64");
			}
			if (constructionPolicy == null) {
				throw new IllegalArgumentException("constructionPolicy must not be null");
			}
			if (integrationOptions == null) {
				throw new IllegalArgumentException("integrationOptions must not be null");
			}
			return new FunctionAnalysisOptions(this);
		}
	}
}
