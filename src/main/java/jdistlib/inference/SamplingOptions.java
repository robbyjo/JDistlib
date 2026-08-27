/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.function.BooleanSupplier;

/** Immutable common MCMC warmup, retention, adaptation, and safety options. */
public final class SamplingOptions {
	private final int warmupIterations;
	private final int sampleIterations;
	private final int thinning;
	private final double stepSize;
	private final double targetAcceptance;
	private final int leapfrogSteps;
	private final int maximumTreeDepth;
	private final double maximumEnergyError;
	private final double sliceWidth;
	private final int maximumSliceSteps;
	private final boolean adaptStepSize;
	private final boolean adaptMassMatrix;
	private final boolean denseMassMatrix;
	private final boolean allowFiniteDifferences;
	private final BooleanSupplier cancellation;

	private SamplingOptions(Builder builder) {
		warmupIterations = builder.warmupIterations;
		sampleIterations = builder.sampleIterations;
		thinning = builder.thinning;
		stepSize = builder.stepSize;
		targetAcceptance = builder.targetAcceptance;
		leapfrogSteps = builder.leapfrogSteps;
		maximumTreeDepth = builder.maximumTreeDepth;
		maximumEnergyError = builder.maximumEnergyError;
		sliceWidth = builder.sliceWidth;
		maximumSliceSteps = builder.maximumSliceSteps;
		adaptStepSize = builder.adaptStepSize;
		adaptMassMatrix = builder.adaptMassMatrix;
		denseMassMatrix = builder.denseMassMatrix;
		allowFiniteDifferences = builder.allowFiniteDifferences;
		cancellation = builder.cancellation;
	}

	public static Builder builder() { return new Builder(); }
	public int warmupIterations() { return warmupIterations; }
	public int sampleIterations() { return sampleIterations; }
	public int thinning() { return thinning; }
	public double stepSize() { return stepSize; }
	public double targetAcceptance() { return targetAcceptance; }
	public int leapfrogSteps() { return leapfrogSteps; }
	public int maximumTreeDepth() { return maximumTreeDepth; }
	public double maximumEnergyError() { return maximumEnergyError; }
	public double sliceWidth() { return sliceWidth; }
	public int maximumSliceSteps() { return maximumSliceSteps; }
	public boolean adaptStepSize() { return adaptStepSize; }
	public boolean adaptMassMatrix() { return adaptMassMatrix; }
	public boolean denseMassMatrix() { return denseMassMatrix; }
	public boolean allowFiniteDifferences() { return allowFiniteDifferences; }
	public boolean cancelled() { return cancellation != null && cancellation.getAsBoolean(); }

	public static final class Builder {
		private int warmupIterations = 1000;
		private int sampleIterations = 1000;
		private int thinning = 1;
		private double stepSize = 0.25;
		private double targetAcceptance = 0.8;
		private int leapfrogSteps = 10;
		private int maximumTreeDepth = 10;
		private double maximumEnergyError = 1000.0;
		private double sliceWidth = 1.0;
		private int maximumSliceSteps = 100;
		private boolean adaptStepSize = true;
		private boolean adaptMassMatrix = true;
		private boolean denseMassMatrix;
		private boolean allowFiniteDifferences;
		private BooleanSupplier cancellation;

		public Builder warmupIterations(int value) { warmupIterations = value; return this; }
		public Builder sampleIterations(int value) { sampleIterations = value; return this; }
		public Builder thinning(int value) { thinning = value; return this; }
		public Builder stepSize(double value) { stepSize = value; return this; }
		public Builder targetAcceptance(double value) { targetAcceptance = value; return this; }
		public Builder leapfrogSteps(int value) { leapfrogSteps = value; return this; }
		public Builder maximumTreeDepth(int value) { maximumTreeDepth = value; return this; }
		public Builder maximumEnergyError(double value) { maximumEnergyError = value; return this; }
		public Builder sliceWidth(double value) { sliceWidth = value; return this; }
		public Builder maximumSliceSteps(int value) { maximumSliceSteps = value; return this; }
		public Builder adaptStepSize(boolean value) { adaptStepSize = value; return this; }
		public Builder adaptMassMatrix(boolean value) { adaptMassMatrix = value; return this; }
		public Builder denseMassMatrix(boolean value) { denseMassMatrix = value; return this; }
		public Builder allowFiniteDifferences(boolean value) { allowFiniteDifferences = value; return this; }
		public Builder cancellation(BooleanSupplier value) { cancellation = value; return this; }

		public SamplingOptions build() {
			if (warmupIterations < 0 || sampleIterations < 1 || thinning < 1
					|| !(stepSize > 0.0) || !Double.isFinite(stepSize)
					|| !(targetAcceptance > 0.0 && targetAcceptance < 1.0)
					|| leapfrogSteps < 1 || maximumTreeDepth < 1 || maximumTreeDepth > 20
					|| !(maximumEnergyError > 0.0)
					|| !(sliceWidth > 0.0) || maximumSliceSteps < 1) {
				throw new IllegalArgumentException("invalid sampling options");
			}
			return new SamplingOptions(this);
		}
	}
}
