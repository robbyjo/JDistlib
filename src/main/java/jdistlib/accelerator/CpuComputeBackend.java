/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Deterministic CPU reference implementation for every accelerated primitive. */
public final class CpuComputeBackend implements ComputeBackend {
	@Override public String id() { return "cpu"; }
	@Override public boolean available() { return true; }
	@Override public ComputeCapabilities capabilities() {
		return new ComputeCapabilities("CPU", System.getProperty("os.arch", "unknown"),
				true, false, 0L, true, true, true, true, false, true, true, true);
	}
	@Override public ComputeDeviceInfo deviceInfo() {
		Package pkg = getClass().getPackage();
		String version = pkg == null || pkg.getImplementationVersion() == null
				? "development" : pkg.getImplementationVersion();
		return new ComputeDeviceInfo(id(), version, ComputeApi.JAVA_CPU,
				System.getProperty("java.version", "unknown"), "n/a",
				System.getProperty("java.vendor", "unknown"),
				System.getProperty("os.name", "unknown"),
				System.getProperty("os.arch", "unknown"), "host", 0L);
	}
	@Override public double[] unary(UnaryOperation operation, double[] input) {
		if (operation == null || input == null) throw new IllegalArgumentException("operation and input required");
		double[] result = new double[input.length];
		for (int i = 0; i < result.length; i++) switch (operation) {
		case EXP: result[i] = Math.exp(input[i]); break;
		case LOG: result[i] = Math.log(input[i]); break;
		case LOG1P: result[i] = Math.log1p(input[i]); break;
		case SQRT: result[i] = Math.sqrt(input[i]); break;
		case TANH: result[i] = Math.tanh(input[i]); break;
		case LOGISTIC: result[i] = logistic(input[i]); break;
		default: throw new AssertionError(operation);
		}
		return result;
	}
	@Override public double[] axpy(double alpha, double[] x, double[] y) {
		checkVectors(x, y); double[] result = new double[x.length];
		for (int i = 0; i < result.length; i++) result[i] = alpha * x[i] + y[i];
		return result;
	}
	@Override public double dot(double[] x, double[] y) {
		checkVectors(x, y); double result = 0.0;
		for (int i = 0; i < x.length; i++) result += x[i] * y[i];
		return result;
	}
	@Override public double[][] matrixMultiply(double[][] left, double[][] right) {
		int[] shape = matrixShape(left), rightShape = matrixShape(right);
		if (shape[1] != rightShape[0]) throw new IllegalArgumentException("matrix dimensions do not conform");
		double[][] result = new double[shape[0]][rightShape[1]];
		for (int i = 0; i < shape[0]; i++) for (int k = 0; k < shape[1]; k++)
			for (int j = 0; j < rightShape[1]; j++) result[i][j] += left[i][k] * right[k][j];
		return result;
	}
	@Override public PreparedTransposeProduct prepareTransposeProduct(double[][] matrix) {
		int[] shape = matrixShape(matrix); final int rows = shape[0], columns = shape[1];
		final double[] values = new double[rows * columns];
		for (int row = 0; row < rows; row++) System.arraycopy(matrix[row], 0, values, row * columns, columns);
		return new PreparedTransposeProduct() {
			@Override public int rows() { return rows; }
			@Override public int columns() { return columns; }
			@Override public double[][] multiply(double[][] vectors) {
				if (vectors == null || vectors.length == 0) throw new IllegalArgumentException("one or more score vectors required");
				double[][] result = new double[vectors.length][columns];
				for (int batch = 0; batch < vectors.length; batch++) {
					double[] vector = vectors[batch]; if (vector == null || vector.length != rows) throw new IllegalArgumentException("score vector length mismatch");
					double[] output = result[batch];
					for (int row = 0; row < rows; row++) { double weight = vector[row]; int offset = row * columns;
						for (int column = 0; column < columns; column++) output[column] += values[offset + column] * weight; }
				}
				return result;
			}
			@Override public void close() {}
		};
	}
	@Override public LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		int[] shape = matrixShape(design);
		if (outcomes == null || outcomes.length != shape[0] || states == null
				|| !(priorPrecision >= 0.0)) throw new IllegalArgumentException("invalid logistic batch");
		double[] values = new double[states.length];
		double[][] gradients = new double[states.length][shape[1]];
		for (int chain = 0; chain < states.length; chain++) {
			if (states[chain].length != shape[1]) throw new IllegalArgumentException("state dimension mismatch");
			for (int d = 0; d < shape[1]; d++) {
				values[chain] -= 0.5 * priorPrecision * states[chain][d] * states[chain][d];
				gradients[chain][d] = -priorPrecision * states[chain][d];
			}
			for (int row = 0; row < shape[0]; row++) {
				double eta = 0.0;
				for (int d = 0; d < shape[1]; d++) eta += design[row][d] * states[chain][d];
				values[chain] += outcomes[row] * eta - log1pExp(eta);
				double residual = outcomes[row] - logistic(eta);
				for (int d = 0; d < shape[1]; d++) gradients[chain][d] += residual * design[row][d];
			}
		}
		return new LogisticRegressionBatchResult(values, gradients);
	}
	@Override public PreparedLogisticRegression prepareLogisticRegression(double[][] design,
			double[] outcomes) {
		int[] shape = matrixShape(design);
		if (outcomes == null || outcomes.length != shape[0])
			throw new IllegalArgumentException("one outcome per row is required");
		final double[][] copiedDesign = new double[shape[0]][];
		for (int i = 0; i < shape[0]; i++) copiedDesign[i] = design[i].clone();
		final double[] copiedOutcomes = outcomes.clone();
		return new PreparedLogisticRegression() {
			@Override public int rows() { return copiedDesign.length; }
			@Override public int dimensions() { return copiedDesign[0].length; }
			@Override public LogisticRegressionBatchResult evaluate(double[][] states,
					double priorPrecision) {
				return logisticRegression(copiedDesign, copiedOutcomes, states, priorPrecision);
			}
			@Override public void close() {}
		};
	}
	@Override public void close() {}
	private static void checkVectors(double[] x, double[] y) {
		if (x == null || y == null || x.length != y.length) throw new IllegalArgumentException("vector lengths must match");
	}
	private static int[] matrixShape(double[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			throw new IllegalArgumentException("matrix must be nonempty");
		int columns = matrix[0].length;
		for (double[] row : matrix) if (row == null || row.length != columns)
			throw new IllegalArgumentException("matrix must be rectangular");
		return new int[] {matrix.length, columns};
	}
	private static double logistic(double value) {
		return value >= 0.0 ? 1.0 / (1.0 + Math.exp(-value))
				: Math.exp(value) / (1.0 + Math.exp(value));
	}
	private static double log1pExp(double value) {
		return value > 0.0 ? value + Math.log1p(Math.exp(-value)) : Math.log1p(Math.exp(value));
	}
}
