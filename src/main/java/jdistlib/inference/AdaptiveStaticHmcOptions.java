/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Controls coordinated ChEES/SNAPER trajectory-length adaptation across chains. */
public final class AdaptiveStaticHmcOptions {
	public enum Criterion { CHEES, SNAPER }
	private final int warmupIterations, sampleIterations, thinning, maximumLeapfrogSteps;
	private final double stepSize, targetAcceptance, maximumEnergyError, jitter;
	private final Criterion criterion; private final MetricConfiguration metric;
	private AdaptiveStaticHmcOptions(Builder builder) { warmupIterations = builder.warmupIterations;
		sampleIterations = builder.sampleIterations; thinning = builder.thinning; maximumLeapfrogSteps = builder.maximumLeapfrogSteps;
		stepSize = builder.stepSize; targetAcceptance = builder.targetAcceptance; maximumEnergyError = builder.maximumEnergyError;
		jitter = builder.jitter; criterion = builder.criterion; metric = builder.metric; }
	public static Builder builder() { return new Builder(); }
	public int warmupIterations() { return warmupIterations; } public int sampleIterations() { return sampleIterations; }
	public int thinning() { return thinning; } public int maximumLeapfrogSteps() { return maximumLeapfrogSteps; }
	public double stepSize() { return stepSize; } public double targetAcceptance() { return targetAcceptance; }
	public double maximumEnergyError() { return maximumEnergyError; } public double jitter() { return jitter; }
	public Criterion criterion() { return criterion; } public MetricConfiguration metric() { return metric; }
	public static final class Builder {
		private int warmupIterations = 1000, sampleIterations = 1000, thinning = 1, maximumLeapfrogSteps = 64;
		private double stepSize = 0.25, targetAcceptance = 0.8, maximumEnergyError = 1000.0, jitter = 0.1;
		private Criterion criterion = Criterion.SNAPER; private MetricConfiguration metric = MetricConfiguration.diagonal();
		public Builder warmupIterations(int value) { warmupIterations = value; return this; }
		public Builder sampleIterations(int value) { sampleIterations = value; return this; }
		public Builder thinning(int value) { thinning = value; return this; }
		public Builder maximumLeapfrogSteps(int value) { maximumLeapfrogSteps = value; return this; }
		public Builder stepSize(double value) { stepSize = value; return this; }
		public Builder targetAcceptance(double value) { targetAcceptance = value; return this; }
		public Builder maximumEnergyError(double value) { maximumEnergyError = value; return this; }
		public Builder jitter(double value) { jitter = value; return this; }
		public Builder criterion(Criterion value) { criterion = value; return this; }
		public Builder metric(MetricConfiguration value) { metric = value; return this; }
		public AdaptiveStaticHmcOptions build() { if (warmupIterations < 1 || sampleIterations < 1 || thinning < 1 || maximumLeapfrogSteps < 1
				|| !(stepSize > 0.0) || !(targetAcceptance > 0.0 && targetAcceptance < 1.0) || !(maximumEnergyError > 0.0)
				|| jitter < 0.0 || jitter > 1.0 || criterion == null || metric == null) throw new IllegalArgumentException("invalid adaptive static HMC options");
			return new AdaptiveStaticHmcOptions(this); }
	}
}
