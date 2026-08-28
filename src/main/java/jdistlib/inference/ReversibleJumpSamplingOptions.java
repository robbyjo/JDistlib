/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.function.BooleanSupplier;

/** Warmup, retention, schedule-adaptation, and streaming options for RJMCMC. */
public final class ReversibleJumpSamplingOptions {
	private final int warmupIterations, sampleIterations, thinning;
	private final boolean adaptMoveWeights, storeDraws;
	private final double targetJumpAcceptance, minimumMoveWeight;
	private final ReversibleJumpProgressListener progress; private final ReversibleJumpDrawSink sink; private final BooleanSupplier cancellation;
	private ReversibleJumpSamplingOptions(Builder builder) {
		warmupIterations = builder.warmupIterations; sampleIterations = builder.sampleIterations; thinning = builder.thinning;
		adaptMoveWeights = builder.adaptMoveWeights; storeDraws = builder.storeDraws; targetJumpAcceptance = builder.targetJumpAcceptance;
		minimumMoveWeight = builder.minimumMoveWeight; progress = builder.progress; sink = builder.sink; cancellation = builder.cancellation;
	}
	public static Builder builder() { return new Builder(); }
	public int warmupIterations() { return warmupIterations; }
	public int sampleIterations() { return sampleIterations; }
	public int thinning() { return thinning; }
	public boolean adaptMoveWeights() { return adaptMoveWeights; }
	public boolean storeDraws() { return storeDraws; }
	public double targetJumpAcceptance() { return targetJumpAcceptance; }
	public double minimumMoveWeight() { return minimumMoveWeight; }
	public boolean cancelled() { return cancellation != null && cancellation.getAsBoolean(); }
	void progress(int completed, int total, boolean warmup, ReversibleJumpIterationStats statistics) { if (progress != null) progress.update(completed, total, warmup, statistics); }
	void emit(int retained, ReversibleJumpState state, double logJoint, ReversibleJumpIterationStats statistics) { if (sink != null) sink.accept(retained, state, logJoint, statistics); }
	public static final class Builder {
		private int warmupIterations = 1000, sampleIterations = 1000, thinning = 1; private boolean adaptMoveWeights = true, storeDraws = true;
		private double targetJumpAcceptance = 0.25, minimumMoveWeight = 1e-3;
		private ReversibleJumpProgressListener progress; private ReversibleJumpDrawSink sink; private BooleanSupplier cancellation;
		private Builder() {}
		public Builder warmupIterations(int value) { warmupIterations = value; return this; }
		public Builder sampleIterations(int value) { sampleIterations = value; return this; }
		public Builder thinning(int value) { thinning = value; return this; }
		public Builder adaptMoveWeights(boolean value) { adaptMoveWeights = value; return this; }
		public Builder targetJumpAcceptance(double value) { targetJumpAcceptance = value; return this; }
		public Builder minimumMoveWeight(double value) { minimumMoveWeight = value; return this; }
		public Builder storeDraws(boolean value) { storeDraws = value; return this; }
		public Builder progressListener(ReversibleJumpProgressListener value) { progress = value; return this; }
		public Builder drawSink(ReversibleJumpDrawSink value) { sink = value; return this; }
		public Builder cancellation(BooleanSupplier value) { cancellation = value; return this; }
		public ReversibleJumpSamplingOptions build() {
			if (warmupIterations < 0 || sampleIterations < 1 || thinning < 1 || !(targetJumpAcceptance > 0.0 && targetJumpAcceptance < 1.0)
					|| !(minimumMoveWeight > 0.0) || !Double.isFinite(minimumMoveWeight)
					|| (long) warmupIterations + (long) sampleIterations * thinning > Integer.MAX_VALUE)
				throw new IllegalArgumentException("invalid RJ sampling options");
			return new ReversibleJumpSamplingOptions(this);
		}
	}
}
