/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Multi-path Gaussian approximation, PSIS diagnostic, and resampled draws. */
public final class PathfinderFit {
	private final double[][] draws, candidates;
	private final double[] logWeights;
	private final OptimizationResult[] optimizations;
	private final int[] selectedPathIterations;
	private final double[] selectedElbos;
	private final double paretoK;
	PathfinderFit(double[][] draws, double[][] candidates, double[] logWeights,
			OptimizationResult[] optimizations, int[] selectedPathIterations,
			double[] selectedElbos, double paretoK) {
		this.draws = copy(draws); this.candidates = copy(candidates);
		this.logWeights = logWeights.clone(); this.optimizations = optimizations.clone();
		this.selectedPathIterations = selectedPathIterations.clone(); this.selectedElbos = selectedElbos.clone(); this.paretoK = paretoK;
	}
	public double[][] draws() { return copy(draws); }
	public double[][] weightedCandidates() { return copy(candidates); }
	public double[] logWeights() { return logWeights.clone(); }
	public OptimizationResult[] optimizations() { return optimizations.clone(); }
	public int[] selectedPathIterations() { return selectedPathIterations.clone(); }
	public double[] selectedElbos() { return selectedElbos.clone(); }
	public double paretoK() { return paretoK; }
	public boolean reliable() { return paretoK <= 0.7; }
	public double[] initialState() { return draws[0].clone(); }
	private static double[][] copy(double[][] values) { double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone(); return result; }
}
