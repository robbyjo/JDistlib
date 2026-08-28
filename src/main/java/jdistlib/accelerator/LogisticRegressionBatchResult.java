/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Batched logistic-regression log densities and gradients. */
public final class LogisticRegressionBatchResult {
	private final double[] logDensities;
	private final double[][] gradients;
	public LogisticRegressionBatchResult(double[] logDensities, double[][] gradients) {
		if (logDensities == null || gradients == null
				|| logDensities.length != gradients.length)
			throw new IllegalArgumentException("batch outputs must match");
		this.logDensities = logDensities.clone();
		this.gradients = new double[gradients.length][];
		for (int i = 0; i < gradients.length; i++) this.gradients[i] = gradients[i].clone();
	}
	public double[] logDensities() { return logDensities.clone(); }
	public double[][] gradients() {
		double[][] result = new double[gradients.length][];
		for (int i = 0; i < result.length; i++) result[i] = gradients[i].clone();
		return result;
	}
}
