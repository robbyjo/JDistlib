/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.function.BooleanSupplier;

import jdistlib.accelerator.Compute;

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
	private final WarmupSchedule warmupSchedule;
	private final MetricConfiguration metricConfiguration;
	private final double integrationTime;
	private final double stepSizeJitter;
	private final ProgressListener progressListener;
	private final DrawSink drawSink;
	private final boolean storeDraws;
	private final BooleanSupplier cancellation;
	private final Compute computeBackend;
	private final ComputeNuts nutsBackend;

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
		warmupSchedule = builder.warmupSchedule;
		metricConfiguration = builder.metricConfiguration != null
				? builder.metricConfiguration : (builder.denseMassMatrix
				? MetricConfiguration.dense() : MetricConfiguration.diagonal());
		integrationTime = builder.integrationTime;
		stepSizeJitter = builder.stepSizeJitter;
		progressListener = builder.progressListener;
		drawSink = builder.drawSink;
		storeDraws = builder.storeDraws;
		cancellation = builder.cancellation;
		computeBackend = builder.computeBackend;
		nutsBackend = builder.nutsBackend;
	}

	public static Builder builder() { return new Builder(); }
	/** Starts a builder that preserves every option, including streaming callbacks. */
	public Builder toBuilder() { return new Builder(this); }
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
	public WarmupSchedule warmupSchedule() { return warmupSchedule; }
	public MetricConfiguration metricConfiguration() { return metricConfiguration; }
	/** Static-HMC integration time; NaN retains the legacy fixed leapfrog count. */
	public double integrationTime() { return integrationTime; }
	public double stepSizeJitter() { return stepSizeJitter; }
	public boolean storeDraws() { return storeDraws; }
	/** Compute policy for accelerator-aware numerical targets and operations. */
	public Compute computeBackend() { return computeBackend; }
	/** NUTS-specific accelerator policy; tree construction always remains on CPU. */
	public ComputeNuts nutsBackend() { return nutsBackend; }
	void progress(int completed, int total, boolean warmup, IterationStats statistics) {
		if (progressListener != null) progressListener.update(completed, total, warmup, statistics);
	}
	void emit(int retained, double[] state, double logDensity, IterationStats statistics) {
		if (drawSink != null) drawSink.accept(retained, state.clone(), logDensity, statistics);
	}
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
		private WarmupSchedule warmupSchedule = WarmupSchedule.stanDefault();
		private MetricConfiguration metricConfiguration;
		private double integrationTime = Double.NaN;
		private double stepSizeJitter;
		private ProgressListener progressListener;
		private DrawSink drawSink;
		private boolean storeDraws = true;
		private BooleanSupplier cancellation;
		private Compute computeBackend = Compute.parse(
				System.getProperty("jdistlib.compute.backend", "auto"));
		private ComputeNuts nutsBackend = ComputeNuts.parse(
				System.getProperty("jdistlib.compute.nuts", "auto"));
		private Builder() {}
		private Builder(SamplingOptions source) {
			warmupIterations = source.warmupIterations; sampleIterations = source.sampleIterations;
			thinning = source.thinning; stepSize = source.stepSize; targetAcceptance = source.targetAcceptance;
			leapfrogSteps = source.leapfrogSteps; maximumTreeDepth = source.maximumTreeDepth;
			maximumEnergyError = source.maximumEnergyError; sliceWidth = source.sliceWidth;
			maximumSliceSteps = source.maximumSliceSteps; adaptStepSize = source.adaptStepSize;
			adaptMassMatrix = source.adaptMassMatrix; denseMassMatrix = source.denseMassMatrix;
			allowFiniteDifferences = source.allowFiniteDifferences; warmupSchedule = source.warmupSchedule;
			metricConfiguration = source.metricConfiguration; integrationTime = source.integrationTime;
			stepSizeJitter = source.stepSizeJitter; progressListener = source.progressListener;
			drawSink = source.drawSink; storeDraws = source.storeDraws; cancellation = source.cancellation;
			computeBackend = source.computeBackend; nutsBackend = source.nutsBackend;
		}

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
		public Builder warmupSchedule(WarmupSchedule value) { warmupSchedule = value; return this; }
		public Builder metric(MetricConfiguration value) { metricConfiguration = value; return this; }
		public Builder integrationTime(double value) { integrationTime = value; return this; }
		public Builder stepSizeJitter(double value) { stepSizeJitter = value; return this; }
		public Builder progressListener(ProgressListener value) { progressListener = value; return this; }
		public Builder drawSink(DrawSink value) { drawSink = value; return this; }
		public Builder storeDraws(boolean value) { storeDraws = value; return this; }
		public Builder cancellation(BooleanSupplier value) { cancellation = value; return this; }
		/** Selects automatic, CPU, or a required accelerator backend. */
		public Builder computeBackend(Compute value) { computeBackend = value; return this; }
		/** Short alias for {@link #computeBackend(Compute)}. */
		public Builder backend(Compute value) { return computeBackend(value); }
		/** Selects off, automatic, or forced NUTS target offload. */
		public Builder nutsBackend(ComputeNuts value) { nutsBackend = value; return this; }

		public SamplingOptions build() {
			if (warmupIterations < 0 || sampleIterations < 1 || thinning < 1
					|| !(stepSize > 0.0) || !Double.isFinite(stepSize)
					|| !(targetAcceptance > 0.0 && targetAcceptance < 1.0)
					|| leapfrogSteps < 1 || maximumTreeDepth < 1 || maximumTreeDepth > 20
					|| !(maximumEnergyError > 0.0) || warmupSchedule == null
					|| (!Double.isNaN(integrationTime) && (!(integrationTime > 0.0)
							|| !Double.isFinite(integrationTime)))
					|| !(stepSizeJitter >= 0.0 && stepSizeJitter <= 1.0)
					|| !(sliceWidth > 0.0) || maximumSliceSteps < 1
					|| computeBackend == null || nutsBackend == null
					|| (computeBackend == Compute.CPU && nutsBackend == ComputeNuts.FORCE)) {
				throw new IllegalArgumentException("invalid sampling options");
			}
			return new SamplingOptions(this);
		}
	}
}
