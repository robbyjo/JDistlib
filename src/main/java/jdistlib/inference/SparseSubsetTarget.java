/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.HashSet;
import java.util.Set;

/** Sparse subset target with an arbitrary candidate count and a bounded active set. */
public final class SparseSubsetTarget {
	private final String[] commonNames, candidateNames; private final int maximumActive; private final SparseSubsetLogJoint logJoint;
	public SparseSubsetTarget(String[] commonParameterNames, String[] candidateNames, int maximumActive,
			SparseSubsetLogJoint logJoint) {
		if (commonParameterNames == null || candidateNames == null || candidateNames.length == 0 || maximumActive < 1
				|| maximumActive > candidateNames.length || logJoint == null) throw new IllegalArgumentException("names, candidates, active bound, and log joint required");
		this.commonNames = commonParameterNames.clone(); this.candidateNames = candidateNames.clone(); this.maximumActive = maximumActive; this.logJoint = logJoint;
		Set<String> names = new HashSet<String>();
		for (String name : this.commonNames) { validateName(name); if (!names.add(name)) throw new IllegalArgumentException("parameter names must be unique"); }
		for (String name : this.candidateNames) { validateName(name); if (!names.add(name)) throw new IllegalArgumentException("parameter names must be unique"); }
	}
	public int candidateCount() { return candidateNames.length; }
	public int commonDimension() { return commonNames.length; }
	public int maximumActive() { return maximumActive; }
	public String candidateName(int candidate) { return candidateNames[candidate]; }
	public String[] candidateNames() { return candidateNames.clone(); }
	public String commonParameterName(int index) { return commonNames[index]; }
	public String[] commonParameterNames() { return commonNames.clone(); }
	public SparseSubsetState state(double[] commonParameters) { return state(new int[0], commonParameters, new double[0]); }
	public SparseSubsetState state(int[] activeCandidates, double[] commonParameters, double[] coefficients) {
		SparseSubsetState state = new SparseSubsetState(activeCandidates, commonParameters, coefficients); validate(state); return state;
	}
	public double logJoint(SparseSubsetState state) {
		validate(state); return logJoint.logJoint(state.commonParameters(), state.activeCandidates(), state.coefficients());
	}
	public void validate(SparseSubsetState state) {
		if (state == null || state.commonDimension() != commonNames.length || state.size() > maximumActive)
			throw new IllegalArgumentException("state does not match sparse target");
		for (int candidate : state.activeCandidates()) if (candidate >= candidateNames.length) throw new IllegalArgumentException("state uses an unknown candidate");
	}
	public String modelName(SparseSubsetState state) {
		validate(state); if (state.size() == 0) return "empty"; StringBuilder out = new StringBuilder();
		for (int candidate : state.activeCandidates()) { if (out.length() > 0) out.append('+'); out.append(candidateNames[candidate]); }
		return out.toString();
	}
	private static void validateName(String name) { if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("parameter names must not be blank"); }
}
