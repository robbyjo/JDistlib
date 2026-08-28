/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Immutable sparse subset, common parameters, and active coefficients. */
public final class SparseSubsetState {
	private final int[] activeCandidates; private final double[] commonParameters, coefficients;
	public SparseSubsetState(int[] activeCandidates, double[] commonParameters, double[] coefficients) {
		if (activeCandidates == null || commonParameters == null || coefficients == null
				|| activeCandidates.length != coefficients.length) throw new IllegalArgumentException("matching sparse state fields required");
		this.activeCandidates = activeCandidates.clone(); this.commonParameters = commonParameters.clone(); this.coefficients = coefficients.clone();
		for (int i = 0; i < this.activeCandidates.length; i++) if (this.activeCandidates[i] < 0
				|| (i > 0 && this.activeCandidates[i - 1] >= this.activeCandidates[i])) throw new IllegalArgumentException("active candidates must be sorted and unique");
		for (double value : this.commonParameters) if (!Double.isFinite(value)) throw new IllegalArgumentException("common parameters must be finite");
		for (double value : this.coefficients) if (!Double.isFinite(value)) throw new IllegalArgumentException("coefficients must be finite");
	}
	public int size() { return activeCandidates.length; }
	public int commonDimension() { return commonParameters.length; }
	public int activeCandidate(int index) { return activeCandidates[index]; }
	public double commonParameter(int index) { return commonParameters[index]; }
	public double coefficient(int index) { return coefficients[index]; }
	public int[] activeCandidates() { return activeCandidates.clone(); }
	public double[] commonParameters() { return commonParameters.clone(); }
	public double[] coefficients() { return coefficients.clone(); }
	public boolean active(int candidate) { return Arrays.binarySearch(activeCandidates, candidate) >= 0; }
	public int activeIndex(int candidate) { return Arrays.binarySearch(activeCandidates, candidate); }
	/** Canonical collision-free textual identity used by sparse model reports. */
	public String modelKey() {
		if (activeCandidates.length == 0) return "empty"; StringBuilder out = new StringBuilder();
		for (int i = 0; i < activeCandidates.length; i++) { if (i > 0) out.append(','); out.append(activeCandidates[i]); }
		return out.toString();
	}
	@Override public boolean equals(Object other) {
		if (!(other instanceof SparseSubsetState)) return false; SparseSubsetState value = (SparseSubsetState) other;
		return Arrays.equals(activeCandidates, value.activeCandidates) && Arrays.equals(commonParameters, value.commonParameters)
				&& Arrays.equals(coefficients, value.coefficients);
	}
	@Override public int hashCode() { return 31 * (31 * Arrays.hashCode(activeCandidates) + Arrays.hashCode(commonParameters)) + Arrays.hashCode(coefficients); }
}
