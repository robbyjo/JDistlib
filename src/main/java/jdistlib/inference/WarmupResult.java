/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable summary of MCMC adaptation. */
public final class WarmupResult {
	private final int iterations;
	private final double initialStepSize;
	private final double finalStepSize;
	private final double[] inverseMassDiagonal;
	private final double[][] inverseMassMatrix;
	private final double meanAcceptanceProbability;

	public WarmupResult(int iterations, double initialStepSize, double finalStepSize,
			double[] inverseMassDiagonal, double meanAcceptanceProbability) {
		this(iterations, initialStepSize, finalStepSize,
				diagonal(inverseMassDiagonal), meanAcceptanceProbability);
	}

	public static WarmupResult withInverseMassMatrix(int iterations,
			double initialStepSize, double finalStepSize, double[][] inverseMassMatrix,
			double meanAcceptanceProbability) {
		return new WarmupResult(iterations, initialStepSize, finalStepSize,
				inverseMassMatrix, meanAcceptanceProbability);
	}

	private WarmupResult(int iterations, double initialStepSize, double finalStepSize,
			double[][] inverseMassMatrix, double meanAcceptanceProbability) {
		this.iterations = iterations;
		this.initialStepSize = initialStepSize;
		this.finalStepSize = finalStepSize;
		this.inverseMassMatrix = copy(inverseMassMatrix);
		if (inverseMassMatrix == null) inverseMassDiagonal = null;
		else {
			inverseMassDiagonal = new double[inverseMassMatrix.length];
			for (int i = 0; i < inverseMassDiagonal.length; i++)
				inverseMassDiagonal[i] = inverseMassMatrix[i][i];
		}
		this.meanAcceptanceProbability = meanAcceptanceProbability;
	}
	public int iterations() { return iterations; }
	public double initialStepSize() { return initialStepSize; }
	public double finalStepSize() { return finalStepSize; }
	public double[] inverseMassDiagonal() {
		return inverseMassDiagonal == null ? null : inverseMassDiagonal.clone();
	}
	public double[][] inverseMassMatrix() { return copy(inverseMassMatrix); }
	public double meanAcceptanceProbability() { return meanAcceptanceProbability; }
	private static double[][] diagonal(double[] values) {
		if (values == null) return null;
		double[][] result = new double[values.length][values.length];
		for (int i = 0; i < values.length; i++) result[i][i] = values[i];
		return result;
	}
	private static double[][] copy(double[][] values) {
		if (values == null) return null;
		double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
		return result;
	}
}
