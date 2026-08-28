/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.PreparedLogisticRegression;

/** Device-resident batched logistic-regression posterior with a spherical normal prior. */
public final class AcceleratedLogisticRegression implements BatchedDifferentiableLogDensity, AutoCloseable {
	private final ComputeBackend backend; private final PreparedLogisticRegression prepared; private final double priorPrecision;
	public AcceleratedLogisticRegression(ComputeBackend backend, double[][] design, double[] outcomes, double priorPrecision) {
		if (backend == null || !backend.available() || !(priorPrecision >= 0.0)) throw new IllegalArgumentException("available backend and nonnegative prior precision required");
		this.backend = backend; this.prepared = backend.prepareLogisticRegression(design, outcomes); this.priorPrecision = priorPrecision;
	}
	@Override public double logDensityAndGradient(double[] state, double[] gradient) {
		if (gradient == null || gradient.length != state.length) throw new IllegalArgumentException("gradient dimension mismatch");
		LogisticRegressionBatchResult result = prepared.evaluate(new double[][] {state}, priorPrecision);
		double[] value = result.gradients()[0]; System.arraycopy(value, 0, gradient, 0, value.length); return result.logDensities()[0];
	}
	@Override public void logDensityAndGradientBatch(double[][] states, double[] logDensities, double[][] gradients) {
		if (states == null || logDensities == null || gradients == null || states.length != logDensities.length || states.length != gradients.length)
			throw new IllegalArgumentException("batch outputs must match states"); LogisticRegressionBatchResult result = prepared.evaluate(states, priorPrecision);
		double[] values = result.logDensities(); double[][] derivatives = result.gradients();
		for (int i = 0; i < states.length; i++) { if (gradients[i] == null || gradients[i].length != states[i].length) throw new IllegalArgumentException("gradient dimension mismatch");
			logDensities[i] = values[i]; System.arraycopy(derivatives[i], 0, gradients[i], 0, derivatives[i].length); }
	}
	public ComputeBackend backend() { return backend; } public int rows() { return prepared.rows(); }
	public int dimensions() { return prepared.dimensions(); } @Override public void close() { prepared.close(); }
}
