/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Common-start grouping required to interpret nested R-hat. */
public final class SuperchainPlan {
	private final double[][] initialStates; private final int chainsPerSuperchain;
	public SuperchainPlan(double[][] initialStates, int chainsPerSuperchain) {
		if (initialStates == null || initialStates.length < 2 || chainsPerSuperchain < 1) throw new IllegalArgumentException("at least two superchains and one subchain are required");
		this.initialStates = new double[initialStates.length][]; int dimension = -1;
		for (int i = 0; i < initialStates.length; i++) { if (initialStates[i] == null || (dimension >= 0 && initialStates[i].length != dimension)) throw new IllegalArgumentException("rectangular initial states required");
			this.initialStates[i] = initialStates[i].clone(); dimension = initialStates[i].length; } this.chainsPerSuperchain = chainsPerSuperchain;
	}
	public int superchains() { return initialStates.length; } public int chainsPerSuperchain() { return chainsPerSuperchain; }
	public int totalChains() { return initialStates.length * chainsPerSuperchain; } public double[][] initialStates() { double[][] result = new double[initialStates.length][];
		for (int i = 0; i < result.length; i++) result[i] = initialStates[i].clone(); return result; }
	double[][] expandedStates() { double[][] result = new double[totalChains()][]; int index = 0; for (double[] initial : initialStates) for (int chain = 0; chain < chainsPerSuperchain; chain++) result[index++] = initial.clone(); return result; }
	int[] ids() { int[] result = new int[totalChains()]; int index = 0; for (int group = 0; group < initialStates.length; group++) for (int chain = 0; chain < chainsPerSuperchain; chain++) result[index++] = group; return result; }
}
