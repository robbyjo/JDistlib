/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.function.BooleanSupplier;

/** Global warmup target and bounded transition segment for restartable sparse RJMCMC. */
public final class SparseSubsetSamplingOptions {
	private final long warmupIterations; private final int segmentTransitions, thinning;
	private final boolean adaptMoveWeights, storeDraws; private final double targetJumpAcceptance, minimumMoveWeight;
	private final SparseSubsetProgressListener progress; private final SparseSubsetDrawSink sink; private final BooleanSupplier cancellation;
	private SparseSubsetSamplingOptions(Builder builder) {
		warmupIterations = builder.warmupIterations; segmentTransitions = builder.segmentTransitions; thinning = builder.thinning;
		adaptMoveWeights = builder.adaptMoveWeights; storeDraws = builder.storeDraws; targetJumpAcceptance = builder.targetJumpAcceptance;
		minimumMoveWeight = builder.minimumMoveWeight; progress = builder.progress; sink = builder.sink; cancellation = builder.cancellation;
	}
	public static Builder builder() { return new Builder(); }
	public long warmupIterations() { return warmupIterations; }
	public int segmentTransitions() { return segmentTransitions; }
	public int thinning() { return thinning; }
	public boolean adaptMoveWeights() { return adaptMoveWeights; }
	public boolean storeDraws() { return storeDraws; }
	public double targetJumpAcceptance() { return targetJumpAcceptance; }
	public double minimumMoveWeight() { return minimumMoveWeight; }
	boolean cancelled() { return cancellation != null && cancellation.getAsBoolean(); }
	void progress(int completed, long total, boolean warmup, SparseSubsetIterationStats stats) { if (progress != null) progress.update(completed, segmentTransitions, total, warmup, stats); }
	void emit(long retained, SparseSubsetState state, double logJoint, SparseSubsetIterationStats stats) { if (sink != null) sink.accept(retained, state, logJoint, stats); }
	public static final class Builder {
		private long warmupIterations = 50000L; private int segmentTransitions = 10000, thinning = 10;
		private boolean adaptMoveWeights = true, storeDraws = true; private double targetJumpAcceptance = 0.25, minimumMoveWeight = 1e-3;
		private SparseSubsetProgressListener progress; private SparseSubsetDrawSink sink; private BooleanSupplier cancellation;
		private Builder() {}
		public Builder warmupIterations(long value) { warmupIterations = value; return this; }
		public Builder segmentTransitions(int value) { segmentTransitions = value; return this; }
		public Builder thinning(int value) { thinning = value; return this; }
		public Builder adaptMoveWeights(boolean value) { adaptMoveWeights = value; return this; }
		public Builder storeDraws(boolean value) { storeDraws = value; return this; }
		public Builder targetJumpAcceptance(double value) { targetJumpAcceptance = value; return this; }
		public Builder minimumMoveWeight(double value) { minimumMoveWeight = value; return this; }
		public Builder progressListener(SparseSubsetProgressListener value) { progress = value; return this; }
		public Builder drawSink(SparseSubsetDrawSink value) { sink = value; return this; }
		public Builder cancellation(BooleanSupplier value) { cancellation = value; return this; }
		public SparseSubsetSamplingOptions build() {
			if (warmupIterations < 0L || segmentTransitions < 1 || thinning < 1
					|| !(targetJumpAcceptance > 0.0 && targetJumpAcceptance < 1.0)
					|| !(minimumMoveWeight > 0.0) || !Double.isFinite(minimumMoveWeight)) throw new IllegalArgumentException("invalid sparse RJ options");
			return new SparseSubsetSamplingOptions(this);
		}
	}
}
