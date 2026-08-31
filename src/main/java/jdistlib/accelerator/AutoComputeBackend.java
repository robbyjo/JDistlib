/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

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
				value.doublePrecision(), value.runtimeCompilation(), value.globalMemoryBytes(),
				value.denseLinearAlgebra(), value.sparseLinearAlgebra(), value.nativeFactorizations());
	}
	@Override public ComputeDeviceInfo deviceInfo() {
		ComputeDeviceInfo value = accelerator.deviceInfo();
		return new ComputeDeviceInfo(id(), value.backendVersion(), ComputeApi.AUTOMATIC,
				value.api().name().toLowerCase(java.util.Locale.ROOT) + " " + value.apiVersion(),
				value.driverVersion(), value.vendor(), value.device(), value.architecture(),
				value.deviceId(), value.globalMemoryBytes());
	}
	@Override public ExecutionPlan plan(LinearAlgebraOperation operation,
			NumericPrecision precision, int... dimensions) {
		if (operation == null || precision == null || dimensions == null)
			throw new IllegalArgumentException("operation, precision, and dimensions are required");
		for (int dimension : dimensions) if (dimension < 0)
			throw new IllegalArgumentException("operation dimensions must be nonnegative");
		ComputeBackend selected = backendFor(operation, dimensions);
		ExecutionPlan concrete = selected.plan(operation, precision, dimensions);
		String reason = operation == LinearAlgebraOperation.CSR_POTRF
				? "sparse-direct factorization uses portable CPU baseline"
				: selected == accelerator ? "work estimate reached AUTO threshold"
				: "work estimate below AUTO threshold";
		return new ExecutionPlan(operation, precision, concrete.kind(), concrete.backendId(),
				concrete.device(), reason);
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
	@Override public void daxpy(int count, double alpha, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride) {
		route(count).daxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
	}
	@Override public double ddot(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		return route(count).ddot(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	@Override public double dnrm2(int count, double[] x, int offset, int stride) {
		return route(count).dnrm2(count, x, offset, stride);
	}
	@Override public void dgemv(MatrixTranspose transpose, int rows, int columns,
			double alpha, double[] matrix, double[] x, double beta, double[] y) {
		long work = saturatingProduct(rows, columns, 1);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu)
				.dgemv(transpose, rows, columns, alpha, matrix, x, beta, y);
	}
	@Override public void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left, double[] right,
			double beta, double[] result) {
		long work = saturatingProduct(rows, columns, shared);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu).dgemm(leftTranspose,
				rightTranspose, rows, columns, shared, alpha, left, right, beta, result);
	}
	@Override public void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, double beta, double[] result) {
		backendFor(LinearAlgebraOperation.SYRK, dimension, dimension, shared)
				.dsyrk(transpose, dimension, shared, alpha, matrix, beta, result);
	}
	@Override public void dtrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		backendFor(LinearAlgebraOperation.TRSV, dimension)
				.dtrsv(triangle, transpose, diagonal, dimension, matrix, vector);
	}
	@Override public void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, double[] right) {
		backendFor(LinearAlgebraOperation.TRSM, rows, columns,
				side == MatrixSide.LEFT ? rows : columns).dtrsm(side, triangle, transpose,
						diagonal, rows, columns, alpha, matrix, right);
	}
	@Override public void dcsrmv(double alpha, CsrMatrix matrix, double[] x,
			double beta, double[] y) {
		long work = matrix == null ? 0L : matrix.nonzeroCount();
		(work >= VECTOR_THRESHOLD ? accelerator : cpu).dcsrmv(alpha, matrix, x, beta, y);
	}
	@Override public void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		long work = matrix == null ? 0L
				: saturatingProduct(matrix.nonzeroCount(), rightColumns, 1);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu)
				.dcsrmm(alpha, matrix, right, rightColumns, beta, result);
	}
	@Override public SparseCholeskyFactor dcsrpotrf(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		return cpu.dcsrpotrf(matrix, triangle, ordering);
	}
	@Override public CholeskyFactor dpotrf(double[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension).dpotrf(matrix, dimension);
	}
	@Override public PivotedQrFactor dgeqp3(double[] matrix, int rows, int columns) {
		return decompositionBackend(rows, columns, Math.min(rows, columns)).dgeqp3(matrix, rows, columns);
	}
	@Override public SymmetricEigenDecomposition dsyev(double[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension).dsyev(matrix, dimension);
	}
	@Override public SingularValueDecomposition dgesvd(double[] matrix, int rows, int columns) {
		return decompositionBackend(rows, columns, Math.min(rows, columns)).dgesvd(matrix, rows, columns);
	}
	@Override public void saxpy(int count, float alpha, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride) {
		route(count).saxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
	}
	@Override public float sdot(int count, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		return route(count).sdot(count, x, xOffset, xStride, y, yOffset, yStride);
	}
	@Override public float snrm2(int count, float[] x, int offset, int stride) {
		return route(count).snrm2(count, x, offset, stride);
	}
	@Override public void sgemv(MatrixTranspose transpose, int rows, int columns,
			float alpha, float[] matrix, float[] x, float beta, float[] y) {
		long work = saturatingProduct(rows, columns, 1);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu)
				.sgemv(transpose, rows, columns, alpha, matrix, x, beta, y);
	}
	@Override public void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left, float[] right,
			float beta, float[] result) {
		long work = saturatingProduct(rows, columns, shared);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu).sgemm(leftTranspose,
				rightTranspose, rows, columns, shared, alpha, left, right, beta, result);
	}
	@Override public void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, float beta, float[] result) {
		backendFor(LinearAlgebraOperation.SYRK, dimension, dimension, shared)
				.ssyrk(transpose, dimension, shared, alpha, matrix, beta, result);
	}
	@Override public void strsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		backendFor(LinearAlgebraOperation.TRSV, dimension)
				.strsv(triangle, transpose, diagonal, dimension, matrix, vector);
	}
	@Override public void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, float[] right) {
		backendFor(LinearAlgebraOperation.TRSM, rows, columns,
				side == MatrixSide.LEFT ? rows : columns).strsm(side, triangle, transpose,
						diagonal, rows, columns, alpha, matrix, right);
	}
	@Override public void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x,
			float beta, float[] y) {
		long work = matrix == null ? 0L : matrix.nonzeroCount();
		(work >= VECTOR_THRESHOLD ? accelerator : cpu).scsrmv(alpha, matrix, x, beta, y);
	}
	@Override public void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		long work = matrix == null ? 0L
				: saturatingProduct(matrix.nonzeroCount(), rightColumns, 1);
		(work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu)
				.scsrmm(alpha, matrix, right, rightColumns, beta, result);
	}
	@Override public FloatSparseCholeskyFactor scsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return cpu.scsrpotrf(matrix, triangle, ordering);
	}
	@Override public FloatCholeskyFactor spotrf(float[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension).spotrf(matrix, dimension);
	}
	@Override public FloatPivotedQrFactor sgeqp3(float[] matrix, int rows, int columns) {
		return decompositionBackend(rows, columns, Math.min(rows, columns)).sgeqp3(matrix, rows, columns);
	}
	@Override public FloatSymmetricEigenDecomposition ssyev(float[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension).ssyev(matrix, dimension);
	}
	@Override public FloatSingularValueDecomposition sgesvd(float[] matrix, int rows, int columns) {
		return decompositionBackend(rows, columns, Math.min(rows, columns)).sgesvd(matrix, rows, columns);
	}
	@Override public PreparedCholesky prepareDpotrf(double[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension)
				.prepareDpotrf(matrix, dimension);
	}
	@Override public PreparedFloatCholesky prepareSpotrf(float[] matrix, int dimension) {
		return decompositionBackend(dimension, dimension, dimension)
				.prepareSpotrf(matrix, dimension);
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
	private ComputeBackend decompositionBackend(int rows, int columns, int iterations) {
		return saturatingProduct(rows, columns, iterations) >= MATRIX_MULTIPLY_THRESHOLD
				? accelerator : cpu;
	}
	private ComputeBackend backendFor(LinearAlgebraOperation operation, int... dimensions) {
		long work;
		switch (operation) {
		case CSR_POTRF:
			return cpu;
		case AXPY: case DOT: case NRM2:
			work = dimensions.length == 0 ? 0L : dimensions[0];
			return work >= VECTOR_THRESHOLD ? accelerator : cpu;
		case GEMV: case TRSV: case CSR_MV:
			work = dimensions.length > 1 ? saturatingProduct(dimensions[0], dimensions[1], 1)
					: dimensions.length == 1 ? saturatingProduct(dimensions[0], dimensions[0], 1) : 0L;
			return work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu;
		default:
			work = dimensions.length >= 3
					? saturatingProduct(dimensions[0], dimensions[1], dimensions[2])
					: dimensions.length == 2 ? saturatingProduct(dimensions[0], dimensions[1], 1)
					: dimensions.length == 1 ? saturatingProduct(dimensions[0], dimensions[0], dimensions[0]) : 0L;
			return work >= MATRIX_MULTIPLY_THRESHOLD ? accelerator : cpu;
		}
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
