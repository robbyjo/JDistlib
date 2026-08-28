/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Unit-Jacobian birth/death mapping that inserts or removes one parameter coordinate. */
public final class CoordinateInsertionTransformation implements DimensionMatchingTransformation {
	private final long smallerModel, largerModel; private final int insertionIndex;
	public CoordinateInsertionTransformation(long smallerModel, long largerModel, int insertionIndex) {
		if (smallerModel < 0L || largerModel < 0L || smallerModel == largerModel || insertionIndex < 0)
			throw new IllegalArgumentException("distinct model ids and insertion index required");
		this.smallerModel = smallerModel; this.largerModel = largerModel; this.insertionIndex = insertionIndex;
	}
	@Override public DimensionMatchingResult forward(ReversibleJumpState state, double[] auxiliary) {
		if (state == null || state.modelId() != smallerModel || auxiliary == null || auxiliary.length != 1
				|| insertionIndex > state.dimension()) throw new IllegalArgumentException("invalid insertion mapping input");
		double[] old = state.parameters(), added = new double[old.length + 1];
		System.arraycopy(old, 0, added, 0, insertionIndex); added[insertionIndex] = auxiliary[0];
		System.arraycopy(old, insertionIndex, added, insertionIndex + 1, old.length - insertionIndex);
		return new DimensionMatchingResult(new ReversibleJumpState(largerModel, added));
	}
	@Override public DimensionMatchingResult inverse(ReversibleJumpState state, double[] auxiliary) {
		if (state == null || state.modelId() != largerModel || auxiliary == null || auxiliary.length != 0
				|| insertionIndex >= state.dimension()) throw new IllegalArgumentException("invalid removal mapping input");
		double[] old = state.parameters(), removed = new double[old.length - 1];
		System.arraycopy(old, 0, removed, 0, insertionIndex);
		System.arraycopy(old, insertionIndex + 1, removed, insertionIndex, old.length - insertionIndex - 1);
		return new DimensionMatchingResult(new ReversibleJumpState(smallerModel, removed), old[insertionIndex]);
	}
	@Override public double logAbsJacobian(ReversibleJumpState state, double[] auxiliary, boolean forward) {
		if (state == null || auxiliary == null) throw new IllegalArgumentException("mapping input required");
		return 0.0;
	}
}
