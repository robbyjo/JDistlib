/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import java.util.Map;
import java.util.TreeMap;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** oneMKL PARDISO analysis, numeric refactorization, and solve handles. */
final class MklPardisoSolver {
	private MklPardisoSolver() {}
	static PreparedSparseCholesky prepare(OneMklComputeBackend backend, CsrMatrix matrix,
			MatrixTriangle triangle) {
		return new DoubleHandle(backend, CanonicalDouble.create(matrix, triangle));
	}
	static PreparedFloatSparseCholesky prepare(OneMklComputeBackend backend,
			FloatCsrMatrix matrix, MatrixTriangle triangle) {
		return new FloatHandle(backend, CanonicalFloat.create(matrix, triangle));
	}

	private abstract static class Handle {
		final Function pardiso, getdiag; final Memory state = new Memory(64L * Native.POINTER_SIZE);
		final int dimension; final int[] rowStarts, columns, permutation, iparm = new int[64];
		private boolean analyzed, closed; int factorNonzeros;
		Handle(OneMklComputeBackend backend, int dimension, int[] rowStarts, int[] columns,
				boolean singlePrecision) {
			this.pardiso = backend.function("pardiso");
			this.getdiag = backend.function("pardiso_getdiag"); this.dimension = dimension;
			this.rowStarts = rowStarts; this.columns = columns; this.permutation = new int[dimension];
			state.clear();
			backend.function("pardisoinit").invokeVoid(new Object[] {state,
					new int[] {2}, iparm});
			iparm[26] = 1; iparm[27] = singlePrecision ? 1 : 0; iparm[34] = 0; iparm[55] = 1;
		}
		final void analyze(Object values) {
			analyzed = true; call(11, values, 1, dummy(), dummy());
		}
		final void numeric(Object values) {
			call(22, values, 1, dummy(), dummy());
			factorNonzeros = iparm[17] > 0 ? iparm[17] : columns.length;
		}
		final void call(int phase, Object values, int rightSides, Object right, Object result) {
			checkOpen(); int[] error = new int[1];
			pardiso.invokeVoid(new Object[] {state, new int[] {1}, new int[] {1},
					new int[] {2}, new int[] {phase}, new int[] {dimension}, values,
					rowStarts, columns, permutation, new int[] {rightSides}, iparm,
					new int[] {0}, right, result, error});
			if (error[0] != 0) throw failure(phase, error[0]);
		}
		final int[] permutationValue() { checkOpen(); return new int[0]; }
		final int factorNonzeroCountValue() { checkOpen(); return factorNonzeros; }
		final void checkOpen() {
			if (closed) throw new IllegalStateException("oneMKL sparse factor is closed");
		}
		final void close(Object values) {
			if (closed) return;
			try { if (analyzed) call(-1, values, 1, dummy(), dummy()); }
			finally { analyzed = false; closed = true; }
		}
		abstract Object dummy();
		private static RuntimeException failure(int phase, int error) {
			String message = "oneMKL PARDISO phase " + phase + " failed with error " + error;
			return error == -4 ? new ArithmeticException(message) : new IllegalStateException(message);
		}
	}

	private static final class DoubleHandle extends Handle implements PreparedSparseCholesky {
		private CanonicalDouble matrix; private double logDeterminant;
		DoubleHandle(OneMklComputeBackend backend, CanonicalDouble matrix) {
			super(backend, matrix.dimension, matrix.rowStarts, matrix.columns, false);
			this.matrix = matrix;
			try { analyze(matrix.values); numeric(matrix.values); updateLogDeterminant(); }
			catch (RuntimeException error) { close(matrix.values); throw error; }
		}
		@Override public int dimension() { checkOpen(); return dimension; }
		@Override public int structuralNonzeroCount() { checkOpen(); return columns.length; }
		@Override public int factorNonzeroCount() { return factorNonzeroCountValue(); }
		@Override public int[] permutation() { return permutationValue(); }
		@Override public double logDeterminant() { checkOpen(); return logDeterminant; }
		@Override public synchronized void refactor(CsrMatrix input) {
			checkOpen(); CanonicalDouble next = CanonicalDouble.create(input, matrix.triangle);
			matrix.requireStructure(next); CanonicalDouble previous = matrix;
			try { numeric(next.values); matrix = next; updateLogDeterminant(); }
			catch (RuntimeException failure) {
				try { numeric(previous.values); matrix = previous; updateLogDeterminant(); }
				catch (RuntimeException restore) { failure.addSuppressed(restore); }
				throw failure;
			}
		}
		@Override public synchronized void solveInPlace(double[] right, int rightColumns) {
			checkOpen(); checkRight(right, rightColumns, dimension);
			double[] nativeRight = toColumnMajor(right, dimension, rightColumns);
			double[] result = new double[right.length];
			call(33, matrix.values, rightColumns, nativeRight, result);
			fromColumnMajor(result, right, dimension, rightColumns);
		}
		@Override public synchronized void close() { close(matrix.values); }
		@Override Object dummy() { return new double[Math.max(1, dimension)]; }
		private void updateLogDeterminant() {
			double[] factorDiagonal = new double[dimension], inputDiagonal = new double[dimension];
			int[] error = new int[1]; getdiag.invokeVoid(new Object[] {state, factorDiagonal,
					inputDiagonal, new int[] {1}, error});
			if (error[0] != 0) throw new IllegalStateException(
					"oneMKL PARDISO diagonal extraction failed with error " + error[0]);
			double sum = 0.0; for (double value : factorDiagonal) {
				if (!(Math.abs(value) > 0.0) || !Double.isFinite(value))
					throw new ArithmeticException("oneMKL PARDISO returned an invalid Cholesky diagonal");
				sum += Math.log(Math.abs(value));
			}
			logDeterminant = sum;
		}
	}

	private static final class FloatHandle extends Handle implements PreparedFloatSparseCholesky {
		private CanonicalFloat matrix; private float logDeterminant;
		FloatHandle(OneMklComputeBackend backend, CanonicalFloat matrix) {
			super(backend, matrix.dimension, matrix.rowStarts, matrix.columns, true);
			this.matrix = matrix;
			try { analyze(matrix.values); numeric(matrix.values); updateLogDeterminant(); }
			catch (RuntimeException error) { close(matrix.values); throw error; }
		}
		@Override public int dimension() { checkOpen(); return dimension; }
		@Override public int structuralNonzeroCount() { checkOpen(); return columns.length; }
		@Override public int factorNonzeroCount() { return factorNonzeroCountValue(); }
		@Override public int[] permutation() { return permutationValue(); }
		@Override public float logDeterminant() { checkOpen(); return logDeterminant; }
		@Override public synchronized void refactor(FloatCsrMatrix input) {
			checkOpen(); CanonicalFloat next = CanonicalFloat.create(input, matrix.triangle);
			matrix.requireStructure(next); CanonicalFloat previous = matrix;
			try { numeric(next.values); matrix = next; updateLogDeterminant(); }
			catch (RuntimeException failure) {
				try { numeric(previous.values); matrix = previous; updateLogDeterminant(); }
				catch (RuntimeException restore) { failure.addSuppressed(restore); }
				throw failure;
			}
		}
		@Override public synchronized void solveInPlace(float[] right, int rightColumns) {
			checkOpen(); checkRight(right, rightColumns, dimension);
			float[] nativeRight = toColumnMajor(right, dimension, rightColumns);
			float[] result = new float[right.length];
			call(33, matrix.values, rightColumns, nativeRight, result);
			fromColumnMajor(result, right, dimension, rightColumns);
		}
		@Override public synchronized void close() { close(matrix.values); }
		@Override Object dummy() { return new float[Math.max(1, dimension)]; }
		private void updateLogDeterminant() {
			float[] factorDiagonal = new float[dimension], inputDiagonal = new float[dimension];
			int[] error = new int[1]; getdiag.invokeVoid(new Object[] {state, factorDiagonal,
					inputDiagonal, new int[] {1}, error});
			if (error[0] != 0) throw new IllegalStateException(
					"oneMKL PARDISO FP32 diagonal extraction failed with error " + error[0]);
			float sum = 0.0f; for (float value : factorDiagonal) {
				if (!(Math.abs(value) > 0.0f) || !Float.isFinite(value))
					throw new ArithmeticException("oneMKL PARDISO returned an invalid FP32 Cholesky diagonal");
				sum += (float) Math.log(Math.abs(value));
			}
			logDeterminant = sum;
		}
	}

	private static final class CanonicalDouble {
		final int dimension; final MatrixTriangle triangle;
		final double[] values; final int[] columns, rowStarts;
		CanonicalDouble(int dimension, MatrixTriangle triangle, double[] values,
				int[] columns, int[] rowStarts) {
			this.dimension = dimension; this.triangle = triangle; this.values = values;
			this.columns = columns; this.rowStarts = rowStarts;
		}
		static CanonicalDouble create(CsrMatrix matrix, MatrixTriangle triangle) {
			if (matrix == null || triangle == null || matrix.rows() < 1
					|| matrix.rows() != matrix.columns())
				throw new IllegalArgumentException("oneMKL sparse Cholesky requires a square matrix and triangle");
			int n = matrix.rows(); TreeMap<Integer, Double>[] rows = doubleRows(n);
			double[] input = matrix.values(); int[] columns = matrix.columnIndices();
			int[] starts = matrix.rowStarts();
			for (int row = 0; row < n; row++) for (int offset = starts[row] - 1;
					offset < starts[row + 1] - 1; offset++) {
				int column = columns[offset] - 1; double value = input[offset];
				if (!Double.isFinite(value)) throw new IllegalArgumentException("sparse matrix must be finite");
				if ((triangle == MatrixTriangle.LOWER && column <= row)
						|| (triangle == MatrixTriangle.UPPER && column >= row)) {
					int upperRow = Math.min(row, column), upperColumn = Math.max(row, column);
					Double previous = rows[upperRow].get(upperColumn);
					rows[upperRow].put(upperColumn, previous == null ? value : previous + value);
				}
			}
			int count = count(rows); double[] values = new double[count];
			int[] resultColumns = new int[count], resultStarts = new int[n + 1];
			int target = 0; resultStarts[0] = 1;
			for (int row = 0; row < n; row++) {
				for (Map.Entry<Integer, Double> entry : rows[row].entrySet()) {
					resultColumns[target] = entry.getKey() + 1; values[target++] = entry.getValue();
				}
				resultStarts[row + 1] = target + 1;
			}
			return new CanonicalDouble(n, triangle, values, resultColumns, resultStarts);
		}
		void requireStructure(CanonicalDouble other) {
			if (dimension != other.dimension || !java.util.Arrays.equals(columns, other.columns)
					|| !java.util.Arrays.equals(rowStarts, other.rowStarts))
				throw new IllegalArgumentException("sparse refactorization structure differs from oneMKL analysis");
		}
	}

	private static final class CanonicalFloat {
		final int dimension; final MatrixTriangle triangle;
		final float[] values; final int[] columns, rowStarts;
		CanonicalFloat(int dimension, MatrixTriangle triangle, float[] values,
				int[] columns, int[] rowStarts) {
			this.dimension = dimension; this.triangle = triangle; this.values = values;
			this.columns = columns; this.rowStarts = rowStarts;
		}
		static CanonicalFloat create(FloatCsrMatrix matrix, MatrixTriangle triangle) {
			if (matrix == null || triangle == null || matrix.rows() < 1
					|| matrix.rows() != matrix.columns())
				throw new IllegalArgumentException("oneMKL FP32 sparse Cholesky requires a square matrix and triangle");
			int n = matrix.rows(); TreeMap<Integer, Float>[] rows = floatRows(n);
			float[] input = matrix.values(); int[] columns = matrix.columnIndices();
			int[] starts = matrix.rowStarts();
			for (int row = 0; row < n; row++) for (int offset = starts[row] - 1;
					offset < starts[row + 1] - 1; offset++) {
				int column = columns[offset] - 1; float value = input[offset];
				if (!Float.isFinite(value)) throw new IllegalArgumentException("FP32 sparse matrix must be finite");
				if ((triangle == MatrixTriangle.LOWER && column <= row)
						|| (triangle == MatrixTriangle.UPPER && column >= row)) {
					int upperRow = Math.min(row, column), upperColumn = Math.max(row, column);
					Float previous = rows[upperRow].get(upperColumn);
					rows[upperRow].put(upperColumn, previous == null ? value : previous + value);
				}
			}
			int count = count(rows); float[] values = new float[count];
			int[] resultColumns = new int[count], resultStarts = new int[n + 1];
			int target = 0; resultStarts[0] = 1;
			for (int row = 0; row < n; row++) {
				for (Map.Entry<Integer, Float> entry : rows[row].entrySet()) {
					resultColumns[target] = entry.getKey() + 1; values[target++] = entry.getValue();
				}
				resultStarts[row + 1] = target + 1;
			}
			return new CanonicalFloat(n, triangle, values, resultColumns, resultStarts);
		}
		void requireStructure(CanonicalFloat other) {
			if (dimension != other.dimension || !java.util.Arrays.equals(columns, other.columns)
					|| !java.util.Arrays.equals(rowStarts, other.rowStarts))
				throw new IllegalArgumentException("FP32 sparse refactorization structure differs from oneMKL analysis");
		}
	}

	@SuppressWarnings("unchecked")
	private static TreeMap<Integer, Double>[] doubleRows(int count) {
		TreeMap<Integer, Double>[] result = (TreeMap<Integer, Double>[]) new TreeMap<?, ?>[count];
		for (int i = 0; i < count; i++) result[i] = new TreeMap<Integer, Double>(); return result;
	}
	@SuppressWarnings("unchecked")
	private static TreeMap<Integer, Float>[] floatRows(int count) {
		TreeMap<Integer, Float>[] result = (TreeMap<Integer, Float>[]) new TreeMap<?, ?>[count];
		for (int i = 0; i < count; i++) result[i] = new TreeMap<Integer, Float>(); return result;
	}
	private static int count(Map<?, ?>[] rows) {
		int result = 0; for (Map<?, ?> row : rows) result += row.size(); return result;
	}
	private static void checkRight(double[] right, int columns, int dimension) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid oneMKL sparse right side");
	}
	private static void checkRight(float[] right, int columns, int dimension) {
		if (columns < 1 || right == null || right.length != dimension * columns)
			throw new IllegalArgumentException("invalid oneMKL FP32 sparse right side");
	}
	private static double[] toColumnMajor(double[] input, int rows, int columns) {
		double[] result = new double[input.length];
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			result[column * rows + row] = input[row * columns + column]; return result;
	}
	private static float[] toColumnMajor(float[] input, int rows, int columns) {
		float[] result = new float[input.length];
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			result[column * rows + row] = input[row * columns + column]; return result;
	}
	private static void fromColumnMajor(double[] input, double[] result, int rows, int columns) {
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			result[row * columns + column] = input[column * rows + row];
	}
	private static void fromColumnMajor(float[] input, float[] result, int rows, int columns) {
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			result[row * columns + column] = input[column * rows + row];
	}
}
