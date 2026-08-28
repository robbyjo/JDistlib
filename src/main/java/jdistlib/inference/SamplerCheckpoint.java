/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Versioned sampler-specific adaptation state for exact resumability. */
public final class SamplerCheckpoint {
	private final String sampler;
	private final int version;
	private final int warmupIteration;
	private final double initialStepSize;
	private final double stepSize;
	private final double[][] inverseMassMatrix;
	private final double[] dualAveragingState;
	private final int covarianceCount;
	private final double[] covarianceMean;
	private final double[][] covarianceProducts;
	private final double warmupAcceptanceSum;

	public SamplerCheckpoint(String sampler, int version, int warmupIteration,
			double initialStepSize, double stepSize, double[][] inverseMassMatrix,
			double[] dualAveragingState, int covarianceCount, double[] covarianceMean,
			double[][] covarianceProducts) {
		this(sampler, version, warmupIteration, initialStepSize, stepSize,
				inverseMassMatrix, dualAveragingState, covarianceCount, covarianceMean,
				covarianceProducts, Double.NaN);
	}

	public SamplerCheckpoint(String sampler, int version, int warmupIteration,
			double initialStepSize, double stepSize, double[][] inverseMassMatrix,
			double[] dualAveragingState, int covarianceCount, double[] covarianceMean,
			double[][] covarianceProducts, double warmupAcceptanceSum) {
		if (sampler == null || version < 1 || warmupIteration < 0)
			throw new IllegalArgumentException("invalid sampler checkpoint");
		this.sampler = sampler; this.version = version;
		this.warmupIteration = warmupIteration;
		this.initialStepSize = initialStepSize; this.stepSize = stepSize;
		this.inverseMassMatrix = copy(inverseMassMatrix);
		this.dualAveragingState = dualAveragingState == null ? null : dualAveragingState.clone();
		this.covarianceCount = covarianceCount;
		this.covarianceMean = covarianceMean == null ? null : covarianceMean.clone();
		this.covarianceProducts = copy(covarianceProducts);
		this.warmupAcceptanceSum = warmupAcceptanceSum;
	}
	public String sampler() { return sampler; }
	public int version() { return version; }
	public int warmupIteration() { return warmupIteration; }
	public double initialStepSize() { return initialStepSize; }
	public double stepSize() { return stepSize; }
	public double[][] inverseMassMatrix() { return copy(inverseMassMatrix); }
	public double[] dualAveragingState() { return dualAveragingState == null ? null : dualAveragingState.clone(); }
	public int covarianceCount() { return covarianceCount; }
	public double[] covarianceMean() { return covarianceMean == null ? null : covarianceMean.clone(); }
	public double[][] covarianceProducts() { return copy(covarianceProducts); }
	public double warmupAcceptanceSum() { return warmupAcceptanceSum; }
	private static double[][] copy(double[][] values) {
		if (values == null) return null;
		double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
		return result;
	}
}
