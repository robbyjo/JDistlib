/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Sparse symbolic and numeric Cholesky baseline without dense materialization. */
final class CpuSparseCholesky {
	private CpuSparseCholesky() {}

	static SparseCholeskyFactor factor(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		check(matrix, triangle, ordering); int n = matrix.rows();
		Map<Integer, Double>[] input = doubleTriangle(matrix, triangle);
		Symbolic symbolic = symbolic(n, graph(input, n), ordering);
		Map<Integer, Double>[] permuted = permute(input, symbolic.inverse, n);
		Map<Integer, Double>[] lower = doubleNumeric(permuted, symbolic.pattern, n);
		int count = count(symbolic.pattern), offset = 0;
		double[] values = new double[count]; int[] columns = new int[count], starts = new int[n + 1];
		starts[0] = 1;
		for (int row = 0; row < n; row++) {
			for (int column : symbolic.pattern[row]) {
				values[offset] = lower[row].get(column); columns[offset++] = column + 1;
			}
			starts[row + 1] = offset + 1;
		}
		return new SparseCholeskyFactor(n, values, columns, starts, symbolic.permutation);
	}

	static FloatSparseCholeskyFactor factor(FloatCsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		check(matrix, triangle, ordering); int n = matrix.rows();
		Map<Integer, Float>[] input = floatTriangle(matrix, triangle);
		Symbolic symbolic = symbolic(n, graphFloat(input, n), ordering);
		Map<Integer, Float>[] permuted = permuteFloat(input, symbolic.inverse, n);
		Map<Integer, Float>[] lower = floatNumeric(permuted, symbolic.pattern, n);
		int count = count(symbolic.pattern), offset = 0;
		float[] values = new float[count]; int[] columns = new int[count], starts = new int[n + 1];
		starts[0] = 1;
		for (int row = 0; row < n; row++) {
			for (int column : symbolic.pattern[row]) {
				values[offset] = lower[row].get(column); columns[offset++] = column + 1;
			}
			starts[row + 1] = offset + 1;
		}
		return new FloatSparseCholeskyFactor(n, values, columns, starts,
				symbolic.permutation);
	}

	private static void check(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		if (matrix == null || matrix.rows() < 1 || matrix.rows() != matrix.columns()
				|| triangle == null || ordering == null)
			throw new IllegalArgumentException("sparse Cholesky requires a square matrix, triangle, and ordering");
	}
	private static void check(FloatCsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		if (matrix == null || matrix.rows() < 1 || matrix.rows() != matrix.columns()
				|| triangle == null || ordering == null)
			throw new IllegalArgumentException("FP32 sparse Cholesky requires a square matrix, triangle, and ordering");
	}

	@SuppressWarnings("unchecked")
	private static Map<Integer, Double>[] doubleTriangle(CsrMatrix matrix,
			MatrixTriangle triangle) {
		int n = matrix.rows();
		Map<Integer, Double>[] result = (Map<Integer, Double>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashMap<Integer, Double>();
		double[] values = matrix.values(); int[] columns = matrix.columnIndices();
		int[] starts = matrix.rowStarts();
		for (int row = 0; row < n; row++) for (int offset = starts[row] - 1;
				offset < starts[row + 1] - 1; offset++) {
			int column = columns[offset] - 1; double value = values[offset];
			if (!Double.isFinite(value)) throw new IllegalArgumentException("sparse matrix must be finite");
			if ((triangle == MatrixTriangle.LOWER && column <= row)
					|| (triangle == MatrixTriangle.UPPER && column >= row)) {
				int lowerRow = Math.max(row, column), lowerColumn = Math.min(row, column);
				Double previous = result[lowerRow].get(lowerColumn);
				result[lowerRow].put(lowerColumn, previous == null ? value : previous + value);
			}
		}
		return result;
	}

	@SuppressWarnings("unchecked")
	private static Map<Integer, Float>[] floatTriangle(FloatCsrMatrix matrix,
			MatrixTriangle triangle) {
		int n = matrix.rows();
		Map<Integer, Float>[] result = (Map<Integer, Float>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashMap<Integer, Float>();
		float[] values = matrix.values(); int[] columns = matrix.columnIndices();
		int[] starts = matrix.rowStarts();
		for (int row = 0; row < n; row++) for (int offset = starts[row] - 1;
				offset < starts[row + 1] - 1; offset++) {
			int column = columns[offset] - 1; float value = values[offset];
			if (!Float.isFinite(value)) throw new IllegalArgumentException("FP32 sparse matrix must be finite");
			if ((triangle == MatrixTriangle.LOWER && column <= row)
					|| (triangle == MatrixTriangle.UPPER && column >= row)) {
				int lowerRow = Math.max(row, column), lowerColumn = Math.min(row, column);
				Float previous = result[lowerRow].get(lowerColumn);
				result[lowerRow].put(lowerColumn, previous == null ? value : previous + value);
			}
		}
		return result;
	}

	@SuppressWarnings("unchecked")
	private static Set<Integer>[] graph(Map<Integer, Double>[] triangle, int n) {
		Set<Integer>[] result = (Set<Integer>[]) new Set<?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashSet<Integer>();
		for (int row = 0; row < n; row++) for (Map.Entry<Integer, Double> entry
				: triangle[row].entrySet()) if (entry.getKey() != row && entry.getValue() != 0.0) {
			result[row].add(entry.getKey()); result[entry.getKey()].add(row);
		}
		return result;
	}
	@SuppressWarnings("unchecked")
	private static Set<Integer>[] graphFloat(Map<Integer, Float>[] triangle, int n) {
		Set<Integer>[] result = (Set<Integer>[]) new Set<?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashSet<Integer>();
		for (int row = 0; row < n; row++) for (Map.Entry<Integer, Float> entry
				: triangle[row].entrySet()) if (entry.getKey() != row && entry.getValue() != 0.0f) {
			result[row].add(entry.getKey()); result[entry.getKey()].add(row);
		}
		return result;
	}

	@SuppressWarnings("unchecked")
	private static Symbolic symbolic(int n, Set<Integer>[] graph, SparseOrdering ordering) {
		boolean[] active = new boolean[n]; for (int i = 0; i < n; i++) active[i] = true;
		int[] permutation = new int[n];
		List<Integer>[] columnRows = (List<Integer>[]) new List<?>[n];
		for (int step = 0; step < n; step++) {
			int selected = ordering == SparseOrdering.NATURAL ? step : minimumDegree(graph, active);
			permutation[step] = selected; active[selected] = false;
			List<Integer> neighbors = new ArrayList<Integer>();
			for (int neighbor : graph[selected]) if (active[neighbor]) neighbors.add(neighbor);
			columnRows[step] = neighbors;
			for (int first = 0; first < neighbors.size(); first++) {
				int left = neighbors.get(first); graph[left].remove(selected);
				for (int second = first + 1; second < neighbors.size(); second++) {
					int right = neighbors.get(second); graph[left].add(right); graph[right].add(left);
				}
			}
			graph[selected].clear();
		}
		int[] inverse = new int[n];
		for (int index = 0; index < n; index++) inverse[permutation[index]] = index;
		TreeSet<Integer>[] pattern = (TreeSet<Integer>[]) new TreeSet<?>[n];
		for (int row = 0; row < n; row++) { pattern[row] = new TreeSet<Integer>(); pattern[row].add(row); }
		for (int column = 0; column < n; column++) for (int oldRow : columnRows[column]) {
			int row = inverse[oldRow];
			if (row <= column) throw new IllegalStateException("invalid sparse elimination ordering");
			pattern[row].add(column);
		}
		return new Symbolic(permutation, inverse, pattern);
	}
	private static int minimumDegree(Set<Integer>[] graph, boolean[] active) {
		int selected = -1, degree = Integer.MAX_VALUE;
		for (int candidate = 0; candidate < active.length; candidate++) if (active[candidate]) {
			int value = 0; for (int neighbor : graph[candidate]) if (active[neighbor]) value++;
			if (value < degree) { selected = candidate; degree = value; }
		}
		return selected;
	}

	@SuppressWarnings("unchecked")
	private static Map<Integer, Double>[] permute(Map<Integer, Double>[] input,
			int[] inverse, int n) {
		Map<Integer, Double>[] result = (Map<Integer, Double>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashMap<Integer, Double>();
		for (int oldRow = 0; oldRow < n; oldRow++) for (Map.Entry<Integer, Double> entry
				: input[oldRow].entrySet()) {
			int first = inverse[oldRow], second = inverse[entry.getKey()];
			int row = Math.max(first, second), column = Math.min(first, second);
			Double previous = result[row].get(column);
			result[row].put(column, previous == null ? entry.getValue()
					: previous + entry.getValue());
		}
		return result;
	}
	@SuppressWarnings("unchecked")
	private static Map<Integer, Float>[] permuteFloat(Map<Integer, Float>[] input,
			int[] inverse, int n) {
		Map<Integer, Float>[] result = (Map<Integer, Float>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) result[row] = new HashMap<Integer, Float>();
		for (int oldRow = 0; oldRow < n; oldRow++) for (Map.Entry<Integer, Float> entry
				: input[oldRow].entrySet()) {
			int first = inverse[oldRow], second = inverse[entry.getKey()];
			int row = Math.max(first, second), column = Math.min(first, second);
			Float previous = result[row].get(column);
			result[row].put(column, previous == null ? entry.getValue()
					: previous + entry.getValue());
		}
		return result;
	}

	@SuppressWarnings("unchecked")
	private static Map<Integer, Double>[] doubleNumeric(Map<Integer, Double>[] matrix,
			TreeSet<Integer>[] pattern, int n) {
		Map<Integer, Double>[] lower = (Map<Integer, Double>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) {
			lower[row] = new HashMap<Integer, Double>();
			for (int column : pattern[row]) {
				Double stored = matrix[row].get(column); double value = stored == null ? 0.0 : stored;
				for (Map.Entry<Integer, Double> entry : lower[row].entrySet()) {
					int previous = entry.getKey(); if (previous >= column) continue;
					Double other = lower[column].get(previous);
					if (other != null) value -= entry.getValue() * other;
				}
				if (row == column) {
					if (!(value > 0.0) || !Double.isFinite(value))
						throw new IllegalArgumentException("sparse matrix is not positive definite at permuted minor " + (row + 1));
					lower[row].put(column, Math.sqrt(value));
				} else lower[row].put(column, value / lower[column].get(column));
			}
		}
		return lower;
	}
	@SuppressWarnings("unchecked")
	private static Map<Integer, Float>[] floatNumeric(Map<Integer, Float>[] matrix,
			TreeSet<Integer>[] pattern, int n) {
		Map<Integer, Float>[] lower = (Map<Integer, Float>[]) new Map<?, ?>[n];
		for (int row = 0; row < n; row++) {
			lower[row] = new HashMap<Integer, Float>();
			for (int column : pattern[row]) {
				Float stored = matrix[row].get(column); float value = stored == null ? 0.0f : stored;
				for (Map.Entry<Integer, Float> entry : lower[row].entrySet()) {
					int previous = entry.getKey(); if (previous >= column) continue;
					Float other = lower[column].get(previous);
					if (other != null) value -= entry.getValue() * other;
				}
				if (row == column) {
					if (!(value > 0.0f) || !Float.isFinite(value))
						throw new IllegalArgumentException("FP32 sparse matrix is not positive definite at permuted minor " + (row + 1));
					lower[row].put(column, (float) Math.sqrt(value));
				} else lower[row].put(column, value / lower[column].get(column));
			}
		}
		return lower;
	}

	private static int count(TreeSet<Integer>[] pattern) {
		int result = 0; for (TreeSet<Integer> row : pattern) result += row.size(); return result;
	}
	private static final class Symbolic {
		final int[] permutation, inverse; final TreeSet<Integer>[] pattern;
		Symbolic(int[] permutation, int[] inverse, TreeSet<Integer>[] pattern) {
			this.permutation = permutation; this.inverse = inverse; this.pattern = pattern;
		}
	}
}
