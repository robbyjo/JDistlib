/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.HashSet;
import java.util.Set;

/** Bit-mask model family for Java-only covariate, locus, or feature selection. */
public final class SubsetSelectionTarget implements ReversibleJumpTarget {
	private final String[] commonNames, candidateNames; private final SubsetLogJoint logJoint; private final long validMask;
	public SubsetSelectionTarget(String[] commonParameterNames, String[] candidateNames, SubsetLogJoint logJoint) {
		if (commonParameterNames == null || candidateNames == null || candidateNames.length == 0 || candidateNames.length > 62 || logJoint == null)
			throw new IllegalArgumentException("common names, one to 62 candidates, and log joint required");
		this.commonNames = commonParameterNames.clone(); this.candidateNames = candidateNames.clone(); this.logJoint = logJoint;
		Set<String> unique = new HashSet<String>();
		for (String name : this.commonNames) { validateName(name); if (!unique.add(name)) throw new IllegalArgumentException("parameter names must be unique"); }
		for (String name : this.candidateNames) { validateName(name); if (!unique.add(name)) throw new IllegalArgumentException("parameter names must be unique"); }
		validMask = (1L << candidateNames.length) - 1L;
	}
	@Override public ReversibleJumpModelSpace modelSpace(long modelId) {
		validateModel(modelId); int[] active = activeCandidates(modelId); String[] names = new String[commonNames.length + active.length];
		System.arraycopy(commonNames, 0, names, 0, commonNames.length);
		for (int i = 0; i < active.length; i++) names[commonNames.length + i] = candidateNames[active[i]];
		return new ReversibleJumpModelSpace(modelId, modelName(modelId), names);
	}
	@Override public double logJoint(ReversibleJumpState state) {
		if (state == null) throw new IllegalArgumentException("state required"); validateModel(state.modelId());
		int[] active = activeCandidates(state.modelId());
		if (state.dimension() != commonNames.length + active.length) return Double.NEGATIVE_INFINITY;
		double[] parameters = state.parameters(), common = new double[commonNames.length], coefficients = new double[active.length];
		System.arraycopy(parameters, 0, common, 0, common.length); System.arraycopy(parameters, common.length, coefficients, 0, coefficients.length);
		return logJoint.logJoint(common, active, coefficients);
	}
	public ReversibleJumpState state(long modelId, double[] commonParameters, double[] activeCoefficients) {
		validateModel(modelId); int count = Long.bitCount(modelId);
		if (commonParameters == null || activeCoefficients == null || commonParameters.length != commonNames.length || activeCoefficients.length != count)
			throw new IllegalArgumentException("parameters do not match subset");
		double[] parameters = new double[commonParameters.length + activeCoefficients.length];
		System.arraycopy(commonParameters, 0, parameters, 0, commonParameters.length);
		System.arraycopy(activeCoefficients, 0, parameters, commonParameters.length, activeCoefficients.length);
		return new ReversibleJumpState(modelId, parameters);
	}
	public int candidateCount() { return candidateNames.length; }
	public int commonDimension() { return commonNames.length; }
	public String candidateName(int candidate) { return candidateNames[candidate]; }
	public String[] candidateNames() { return candidateNames.clone(); }
	public String[] commonParameterNames() { return commonNames.clone(); }
	public boolean active(long modelId, int candidate) { validateModel(modelId); return candidate >= 0 && candidate < candidateNames.length && (modelId & 1L << candidate) != 0L; }
	public int[] activeCandidates(long modelId) {
		validateModel(modelId); int[] result = new int[Long.bitCount(modelId)]; int offset = 0;
		for (int candidate = 0; candidate < candidateNames.length; candidate++) if ((modelId & 1L << candidate) != 0L) result[offset++] = candidate;
		return result;
	}
	public int parameterIndex(long modelId, int candidate) {
		if (!active(modelId, candidate)) throw new IllegalArgumentException("candidate is not active");
		return commonNames.length + Long.bitCount(modelId & ((1L << candidate) - 1L));
	}
	public String modelName(long modelId) {
		validateModel(modelId); if (modelId == 0L) return "intercept-only"; StringBuilder result = new StringBuilder();
		for (int candidate : activeCandidates(modelId)) { if (result.length() > 0) result.append('+'); result.append(candidateNames[candidate]); }
		return result.toString();
	}
	private void validateModel(long modelId) { if (modelId < 0L || (modelId & ~validMask) != 0L) throw new IllegalArgumentException("model id uses an unknown candidate"); }
	private static void validateName(String name) { if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("parameter names must not be blank"); }
}
