/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Immutable model identifier and its dimension-specific parameter vector. */
public final class ReversibleJumpState {
	private final long modelId; private final double[] parameters;
	public ReversibleJumpState(long modelId, double... parameters) {
		if (modelId < 0L || parameters == null) throw new IllegalArgumentException("nonnegative model id and parameters required");
		this.modelId = modelId; this.parameters = parameters.clone();
		for (double parameter : this.parameters) if (!Double.isFinite(parameter)) throw new IllegalArgumentException("RJ parameters must be finite");
	}
	public long modelId() { return modelId; }
	public int dimension() { return parameters.length; }
	public double parameter(int index) { return parameters[index]; }
	public double[] parameters() { return parameters.clone(); }
	@Override public boolean equals(Object other) {
		if (!(other instanceof ReversibleJumpState)) return false;
		ReversibleJumpState value = (ReversibleJumpState) other;
		return modelId == value.modelId && Arrays.equals(parameters, value.parameters);
	}
	@Override public int hashCode() { return 31 * Long.valueOf(modelId).hashCode() + Arrays.hashCode(parameters); }
}
