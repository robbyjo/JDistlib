/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Per-iteration sampler statistics used by convergence diagnostics. */
public final class IterationStats {
	private final boolean accepted;
	private final double acceptanceProbability;
	private final double stepSize;
	private final double energy;
	private final double energyError;
	private final boolean divergent;
	private final int treeDepth;
	private final boolean treeDepthSaturated;
	private final int leapfrogSteps;

	public IterationStats(boolean accepted, double acceptanceProbability,
			double stepSize, double energy, double energyError, boolean divergent,
			int treeDepth, int leapfrogSteps) {
		this(accepted, acceptanceProbability, stepSize, energy, energyError,
				divergent, treeDepth, false, leapfrogSteps);
	}

	public IterationStats(boolean accepted, double acceptanceProbability,
			double stepSize, double energy, double energyError, boolean divergent,
			int treeDepth, boolean treeDepthSaturated, int leapfrogSteps) {
		this.accepted = accepted;
		this.acceptanceProbability = acceptanceProbability;
		this.stepSize = stepSize;
		this.energy = energy;
		this.energyError = energyError;
		this.divergent = divergent;
		this.treeDepth = treeDepth;
		this.treeDepthSaturated = treeDepthSaturated;
		this.leapfrogSteps = leapfrogSteps;
	}
	public boolean accepted() { return accepted; }
	public double acceptanceProbability() { return acceptanceProbability; }
	public double stepSize() { return stepSize; }
	public double energy() { return energy; }
	public double energyError() { return energyError; }
	public boolean divergent() { return divergent; }
	public int treeDepth() { return treeDepth; }
	public boolean treeDepthSaturated() { return treeDepthSaturated; }
	public int leapfrogSteps() { return leapfrogSteps; }
}
