/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Optional backend for vector, dense-linear-algebra, and batched likelihood work. */
public interface ComputeBackend extends AutoCloseable {
	String id();
	boolean available();
	ComputeCapabilities capabilities();
	double[] unary(UnaryOperation operation, double[] input);
	double[] axpy(double alpha, double[] x, double[] y);
	double dot(double[] x, double[] y);
	double[][] matrixMultiply(double[][] left, double[][] right);
	LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision);
	/** Keeps reusable data in backend-optimal storage when the backend supports it. */
	default PreparedLogisticRegression prepareLogisticRegression(final double[][] design,
			final double[] outcomes) {
		return new PreparedLogisticRegression() {
			@Override public int rows() { return design.length; }
			@Override public int dimensions() { return design[0].length; }
			@Override public LogisticRegressionBatchResult evaluate(double[][] states,
					double priorPrecision) {
				return logisticRegression(design, outcomes, states, priorPrecision);
			}
			@Override public void close() {}
		};
	}
	@Override void close();
}
