/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Split/merge map x,u to x+u,x-u with forward log-Jacobian log(2). */
public final class CoordinateSplitTransformation implements DimensionMatchingTransformation {
	private final long mergedModel, splitModel; private final int coordinate;
	public CoordinateSplitTransformation(long mergedModel, long splitModel, int coordinate) {
		if (mergedModel < 0L || splitModel < 0L || mergedModel == splitModel || coordinate < 0)
			throw new IllegalArgumentException("distinct model ids and split coordinate required");
		this.mergedModel = mergedModel; this.splitModel = splitModel; this.coordinate = coordinate;
	}
	@Override public DimensionMatchingResult forward(ReversibleJumpState state, double[] auxiliary) {
		if (state == null || state.modelId() != mergedModel || coordinate >= state.dimension() || auxiliary == null || auxiliary.length != 1)
			throw new IllegalArgumentException("invalid split mapping input");
		double[] old = state.parameters(), split = new double[old.length + 1];
		System.arraycopy(old, 0, split, 0, coordinate); split[coordinate] = old[coordinate] + auxiliary[0];
		split[coordinate + 1] = old[coordinate] - auxiliary[0];
		System.arraycopy(old, coordinate + 1, split, coordinate + 2, old.length - coordinate - 1);
		return new DimensionMatchingResult(new ReversibleJumpState(splitModel, split));
	}
	@Override public DimensionMatchingResult inverse(ReversibleJumpState state, double[] auxiliary) {
		if (state == null || state.modelId() != splitModel || coordinate + 1 >= state.dimension() || auxiliary == null || auxiliary.length != 0)
			throw new IllegalArgumentException("invalid merge mapping input");
		double[] old = state.parameters(), merged = new double[old.length - 1];
		System.arraycopy(old, 0, merged, 0, coordinate); merged[coordinate] = 0.5 * (old[coordinate] + old[coordinate + 1]);
		System.arraycopy(old, coordinate + 2, merged, coordinate + 1, old.length - coordinate - 2);
		return new DimensionMatchingResult(new ReversibleJumpState(mergedModel, merged),
				0.5 * (old[coordinate] - old[coordinate + 1]));
	}
	@Override public double logAbsJacobian(ReversibleJumpState state, double[] auxiliary, boolean forward) {
		if (state == null || auxiliary == null) throw new IllegalArgumentException("mapping input required"); return forward ? Math.log(2.0) : -Math.log(2.0);
	}
}
