/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Posterior summary and modern multi-chain convergence diagnostics. */
public final class ParameterDiagnostics {
	private final String name;
	private final double mean;
	private final double standardDeviation;
	private final double median;
	private final double lowerQuantile;
	private final double upperQuantile;
	private final double rHat;
	private final double bulkEffectiveSampleSize;
	private final double tailEffectiveSampleSize;
	private final double monteCarloStandardError;
	private final boolean reliable;

	ParameterDiagnostics(String name, double mean, double standardDeviation,
			double median, double lowerQuantile, double upperQuantile, double rHat,
			double bulkEffectiveSampleSize, double tailEffectiveSampleSize,
			double monteCarloStandardError, boolean reliable) {
		this.name = name; this.mean = mean; this.standardDeviation = standardDeviation;
		this.median = median; this.lowerQuantile = lowerQuantile;
		this.upperQuantile = upperQuantile; this.rHat = rHat;
		this.bulkEffectiveSampleSize = bulkEffectiveSampleSize;
		this.tailEffectiveSampleSize = tailEffectiveSampleSize;
		this.monteCarloStandardError = monteCarloStandardError;
		this.reliable = reliable;
	}
	public String name() { return name; }
	public double mean() { return mean; }
	public double standardDeviation() { return standardDeviation; }
	public double median() { return median; }
	public double lowerQuantile() { return lowerQuantile; }
	public double upperQuantile() { return upperQuantile; }
	public double rHat() { return rHat; }
	public double bulkEffectiveSampleSize() { return bulkEffectiveSampleSize; }
	public double tailEffectiveSampleSize() { return tailEffectiveSampleSize; }
	public double monteCarloStandardError() { return monteCarloStandardError; }
	public boolean reliable() { return reliable; }
}
