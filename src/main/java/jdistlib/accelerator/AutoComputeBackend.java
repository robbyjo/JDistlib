/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/**
 * Size-aware CPU/accelerator router used by {@link Compute#AUTO}.
 * Thresholds are deliberately conservative because inputs originate on the JVM heap.
 */
final class AutoComputeBackend implements ComputeBackend {
	static final int VECTOR_THRESHOLD = 32768;
	static final long MATRIX_MULTIPLY_THRESHOLD = 1000000L;
	static final long LOGISTIC_THRESHOLD = 1000000L;
	private final ComputeBackend cpu, accelerator;
	AutoComputeBackend(ComputeBackend cpu, ComputeBackend accelerator) {
		this.cpu = cpu; this.accelerator = accelerator;
	}
	String acceleratorId() { return accelerator.id(); }
	@Override public String id() { return "auto"; }
	@Override public String selectedBackend() { return accelerator.id(); }
	@Override public boolean automaticRouting() { return true; }
	@Override public boolean available() { return cpu.available() && accelerator.available(); }
	@Override public ComputeCapabilities capabilities() {
		ComputeCapabilities value = accelerator.capabilities();
		return new ComputeCapabilities("AUTO(" + value.backend() + ")", value.device(),
				value.doublePrecision(), value.runtimeCompilation(), value.globalMemoryBytes());
	}
	@Override public double[] unary(UnaryOperation operation, double[] input) {
		return route(input == null ? 0 : input.length).unary(operation, input);
	}
	@Override public double[] axpy(double alpha, double[] x, double[] y) {
		return route(x == null ? 0 : x.length).axpy(alpha, x, y);
	}
	@Override public double dot(double[] x, double[] y) {
		return route(x == null ? 0 : x.length).dot(x, y);
	}
	@Override public double[][] matrixMultiply(double[][] left, double[][] right) {
		long work = 0L;
		if (left != null && left.length > 0 && left[0] != null && right != null
				&& right.length > 0 && right[0] != null)
			work = saturatingProduct(left.length, left[0].length, right[0].length);
		return (work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu)
				.matrixMultiply(left, right);
	}
	@Override public PreparedTransposeProduct prepareTransposeProduct(final double[][] matrix) {
		final PreparedTransposeProduct cpuPrepared = cpu.prepareTransposeProduct(matrix);
		final PreparedTransposeProduct acceleratorPrepared;
		try { acceleratorPrepared = accelerator.prepareTransposeProduct(matrix); }
		catch (RuntimeException failure) { cpuPrepared.close(); throw failure; }
		return new PreparedTransposeProduct() {
			@Override public int rows() { return cpuPrepared.rows(); }
			@Override public int columns() { return cpuPrepared.columns(); }
			@Override public double[][] multiply(double[][] vectors) {
				if (vectors == null) throw new IllegalArgumentException("score vectors required");
				long work = saturatingProduct(rows(), columns(), vectors.length);
				return (work >= LOGISTIC_THRESHOLD ? acceleratorPrepared : cpuPrepared).multiply(vectors);
			}
			@Override public void close() { try { acceleratorPrepared.close(); } finally { cpuPrepared.close(); } }
		};
	}
	@Override public LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		long work = logisticWork(design, states);
		return (work >= LOGISTIC_THRESHOLD ? accelerator : cpu)
				.logisticRegression(design, outcomes, states, priorPrecision);
	}
	@Override public PreparedLogisticRegression prepareLogisticRegression(
			double[][] design, double[] outcomes) {
		final PreparedLogisticRegression cpuPrepared = cpu.prepareLogisticRegression(design, outcomes);
		final PreparedLogisticRegression acceleratorPrepared = accelerator.prepareLogisticRegression(design, outcomes);
		return new PreparedLogisticRegression() {
			@Override public int rows() { return cpuPrepared.rows(); }
			@Override public int dimensions() { return cpuPrepared.dimensions(); }
			@Override public LogisticRegressionBatchResult evaluate(double[][] states,
					double priorPrecision) {
				long work = saturatingProduct(rows(), dimensions(),
						states == null ? 0 : states.length);
				return (work >= LOGISTIC_THRESHOLD ? acceleratorPrepared : cpuPrepared)
						.evaluate(states, priorPrecision);
			}
			@Override public void close() {
				try { acceleratorPrepared.close(); } finally { cpuPrepared.close(); }
			}
		};
	}
	private ComputeBackend route(int length) {
		return length >= VECTOR_THRESHOLD ? accelerator : cpu;
	}
	private static long logisticWork(double[][] design, double[][] states) {
		return design == null || design.length == 0 || design[0] == null
				? 0L : saturatingProduct(design.length, design[0].length,
						states == null ? 0 : states.length);
	}
	private static long saturatingProduct(int first, int second, int third) {
		if (first <= 0 || second <= 0 || third <= 0) return 0L;
		long value = (long) first * second;
		return value > Long.MAX_VALUE / third ? Long.MAX_VALUE : value * third;
	}
	@Override public void close() {
		try { accelerator.close(); } finally { cpu.close(); }
	}
}
