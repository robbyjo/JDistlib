/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.math.IntegrationOptions;

/** Immutable settings for probability-kernel sanity analysis. */
public final class FunctionAnalysisOptions {
	private final int sampleCount;
	private final int repeatabilityChecks;
	private final double discontinuityRatio;
	private final double dynamicRangeOrders;
	private final IntegrationOptions integrationOptions;

	private FunctionAnalysisOptions(Builder builder) {
		sampleCount = builder.sampleCount;
		repeatabilityChecks = builder.repeatabilityChecks;
		discontinuityRatio = builder.discontinuityRatio;
		dynamicRangeOrders = builder.dynamicRangeOrders;
		integrationOptions = builder.integrationOptions;
	}

	public static Builder builder() { return new Builder(); }
	public static FunctionAnalysisOptions defaults() { return builder().build(); }
	public int getSampleCount() { return sampleCount; }
	public int getRepeatabilityChecks() { return repeatabilityChecks; }
	public double getDiscontinuityRatio() { return discontinuityRatio; }
	public double getDynamicRangeOrders() { return dynamicRangeOrders; }
	public IntegrationOptions getIntegrationOptions() { return integrationOptions; }

	public static final class Builder {
		private int sampleCount = 257;
		private int repeatabilityChecks = 5;
		private double discontinuityRatio = 1e6;
		private double dynamicRangeOrders = 12.0;
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
			if (integrationOptions == null) {
				throw new IllegalArgumentException("integrationOptions must not be null");
			}
			return new FunctionAnalysisOptions(this);
		}
	}
}
