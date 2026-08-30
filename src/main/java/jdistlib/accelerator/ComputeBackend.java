/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Optional backend for vector, dense-linear-algebra, and batched likelihood work. */
public interface ComputeBackend extends LinearAlgebraBackend,
		SinglePrecisionLinearAlgebraBackend, AutoCloseable {
	String id();
	/** Concrete provider used for accelerated work; differs from {@link #id()} for AUTO routing. */
	default String selectedBackend() { return id(); }
	/** Whether individual operations may route between CPU and the selected provider. */
	default boolean automaticRouting() { return false; }
	boolean available();
	ComputeCapabilities capabilities();
	default double[] unary(UnaryOperation operation, double[] input) {
		return new CpuComputeBackend().unary(operation, input);
	}
	default double[] axpy(double alpha, double[] x, double[] y) {
		if (x == null || y == null || x.length != y.length)
			throw new IllegalArgumentException("vector lengths must match");
		double[] result = y.clone(); daxpy(x.length, alpha, x, 0, 1, result, 0, 1); return result;
	}
	default double dot(double[] x, double[] y) {
		if (x == null || y == null || x.length != y.length)
			throw new IllegalArgumentException("vector lengths must match");
		return ddot(x.length, x, 0, 1, y, 0, 1);
	}
	default double[][] matrixMultiply(double[][] left, double[][] right) {
		if (left == null || right == null || left.length == 0 || right.length == 0
				|| left[0] == null || right[0] == null || left[0].length != right.length)
			throw new IllegalArgumentException("matrix dimensions do not conform");
		int rows = left.length, shared = right.length, columns = right[0].length;
		double[] a = new double[rows * shared], b = new double[shared * columns], c = new double[rows * columns];
		for (int i = 0; i < rows; i++) {
			if (left[i] == null || left[i].length != shared) throw new IllegalArgumentException("left matrix must be rectangular");
			System.arraycopy(left[i], 0, a, i * shared, shared);
		}
		for (int i = 0; i < shared; i++) {
			if (right[i] == null || right[i].length != columns) throw new IllegalArgumentException("right matrix must be rectangular");
			System.arraycopy(right[i], 0, b, i * columns, columns);
		}
		dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE, rows, columns, shared, 1.0, a, b, 0.0, c);
		double[][] result = new double[rows][columns];
		for (int i = 0; i < rows; i++) System.arraycopy(c, i * columns, result[i], 0, columns);
		return result;
	}
	/** Keeps a reusable row-by-feature matrix ready for repeated {@code X'v} batches. */
	default PreparedTransposeProduct prepareTransposeProduct(final double[][] matrix) {
		final CpuComputeBackend cpu = new CpuComputeBackend();
		return cpu.prepareTransposeProduct(matrix);
	}
	default LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		return new CpuComputeBackend().logisticRegression(design, outcomes, states, priorPrecision);
	}
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
