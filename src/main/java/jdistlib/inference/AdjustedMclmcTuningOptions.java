/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Pilot-search controls for adjusted MCLMC step size and decorrelation length. */
public final class AdjustedMclmcTuningOptions {
	private final int maximumLeapfrogSteps, pilotWarmup, pilotDraws;
	private final double targetAcceptance;
	private AdjustedMclmcTuningOptions(Builder builder) { maximumLeapfrogSteps = builder.maximumLeapfrogSteps;
		pilotWarmup = builder.pilotWarmup; pilotDraws = builder.pilotDraws; targetAcceptance = builder.targetAcceptance; }
	public static Builder builder() { return new Builder(); }
	public int maximumLeapfrogSteps() { return maximumLeapfrogSteps; } public int pilotWarmup() { return pilotWarmup; }
	public int pilotDraws() { return pilotDraws; } public double targetAcceptance() { return targetAcceptance; }
	public static final class Builder {
		private int maximumLeapfrogSteps = 64, pilotWarmup = 100, pilotDraws = 100; private double targetAcceptance = 0.9;
		public Builder maximumLeapfrogSteps(int value) { maximumLeapfrogSteps = value; return this; }
		public Builder pilotWarmup(int value) { pilotWarmup = value; return this; }
		public Builder pilotDraws(int value) { pilotDraws = value; return this; }
		public Builder targetAcceptance(double value) { targetAcceptance = value; return this; }
		public AdjustedMclmcTuningOptions build() { if (maximumLeapfrogSteps < 1 || pilotWarmup < 1 || pilotDraws < 4
				|| !(targetAcceptance > 0.0 && targetAcceptance < 1.0)) throw new IllegalArgumentException("invalid adjusted MCLMC tuning options"); return new AdjustedMclmcTuningOptions(this); }
	}
}
