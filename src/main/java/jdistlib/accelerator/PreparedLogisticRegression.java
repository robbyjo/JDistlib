/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** A logistic-regression data set prepared for repeated batched evaluation. */
public interface PreparedLogisticRegression extends AutoCloseable {
	int rows();
	int dimensions();
	LogisticRegressionBatchResult evaluate(double[][] states, double priorPrecision);
	@Override void close();
}
