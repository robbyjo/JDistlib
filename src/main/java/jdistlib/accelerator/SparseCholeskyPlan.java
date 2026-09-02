/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/**
 * Provider-facing symbolic sparse-Cholesky plan.
 *
 * <p>The plan owns a fill-reducing symmetric permutation and the zero-based CSR
 * pattern of the resulting lower factor.  Native providers use
 * {@link #factorValues(CsrMatrix)} or {@link #factorValues(FloatCsrMatrix)} to
 * scatter a new authoritative input triangle into that fixed pattern before a
 * numerical refactorization.  Symbolic analysis therefore remains reusable
 * without exposing provider-specific factor storage in the public factor API.</p>
 */
public final class SparseCholeskyPlan {
	private final int dimension, structuralNonzeros;
	private final MatrixTriangle triangle;
	private final int[] permutation, inverse, columnIndices, rowStarts;
	private final long[] structure;

	/** Analyzes an FP64 sparse SPD structure without performing numeric factorization. */
	public static SparseCholeskyPlan analyze(CsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		Canonical canonical = Canonical.create(matrix, triangle);
		return analyze(canonical, triangle, ordering);
	}

	/** Analyzes an FP32 sparse SPD structure without performing numeric factorization. */
	public static SparseCholeskyPlan analyze(FloatCsrMatrix matrix, MatrixTriangle triangle,
			SparseOrdering ordering) {
		Canonical canonical = Canonical.create(matrix, triangle);
		return analyze(canonical, triangle, ordering);
	}

	private static SparseCholeskyPlan analyze(Canonical canonical, MatrixTriangle triangle,
			SparseOrdering ordering) {
		if (ordering == null) throw new IllegalArgumentException("sparse ordering is required");
		int n = canonical.dimension;
		@SuppressWarnings("unchecked") Set<Integer>[] graph = (Set<Integer>[]) new Set<?>[n];
		for (int row = 0; row < n; row++) graph[row] = new HashSet<Integer>();
		for (long key : canonical.keys) {
			int row = row(key), column = column(key);
			if (row != column) { graph[row].add(column); graph[column].add(row); }
		}
		boolean[] active = new boolean[n]; Arrays.fill(active, true);
		int[] permutation = new int[n];
		@SuppressWarnings("unchecked") List<Integer>[] columnRows = (List<Integer>[]) new List<?>[n];
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
		@SuppressWarnings("unchecked") TreeSet<Integer>[] pattern = (TreeSet<Integer>[]) new TreeSet<?>[n];
		for (int row = 0; row < n; row++) { pattern[row] = new TreeSet<Integer>(); pattern[row].add(row); }
		for (int factorColumn = 0; factorColumn < n; factorColumn++)
			for (int oldRow : columnRows[factorColumn]) {
				int factorRow = inverse[oldRow];
				if (factorRow <= factorColumn) throw new IllegalStateException("invalid sparse elimination ordering");
				pattern[factorRow].add(factorColumn);
			}
		int count = 0; for (TreeSet<Integer> row : pattern) count += row.size();
		int[] columns = new int[count], starts = new int[n + 1]; int offset = 0;
		for (int row = 0; row < n; row++) {
			starts[row] = offset;
			for (int column : pattern[row]) columns[offset++] = column;
		}
		starts[n] = offset;
		return new SparseCholeskyPlan(n, canonical.keys.length, triangle, permutation,
				inverse, columns, starts, canonical.keys);
	}

	private SparseCholeskyPlan(int dimension, int structuralNonzeros,
			MatrixTriangle triangle, int[] permutation, int[] inverse,
			int[] columnIndices, int[] rowStarts, long[] structure) {
		this.dimension = dimension; this.structuralNonzeros = structuralNonzeros;
		this.triangle = triangle; this.permutation = permutation; this.inverse = inverse;
		this.columnIndices = columnIndices; this.rowStarts = rowStarts;
		this.structure = structure;
	}

	public int dimension() { return dimension; }
	public int structuralNonzeroCount() { return structuralNonzeros; }
	public int factorNonzeroCount() { return columnIndices.length; }
	/** Returns the new-to-original symmetric permutation. */
	public int[] permutation() { return permutation.clone(); }
	/** Returns zero-based lower-factor CSR column indices. */
	public int[] factorColumnIndices() { return columnIndices.clone(); }
	/** Returns zero-based lower-factor CSR row starts. */
	public int[] factorRowStarts() { return rowStarts.clone(); }

	/** Scatters an FP64 authoritative triangle into the fixed factor pattern. */
	public double[] factorValues(CsrMatrix matrix) {
		Canonical canonical = Canonical.create(matrix, triangle);
		requireStructure(canonical); double[] result = new double[columnIndices.length];
		for (int source = 0; source < canonical.keys.length; source++) {
			long key = canonical.keys[source]; int first = inverse[row(key)], second = inverse[column(key)];
			int factorRow = Math.max(first, second), factorColumn = Math.min(first, second);
			result[offset(factorRow, factorColumn)] = canonical.doubleValues[source];
		}
		return result;
	}

	/** Scatters an FP32 authoritative triangle into the fixed factor pattern. */
	public float[] factorValues(FloatCsrMatrix matrix) {
		Canonical canonical = Canonical.create(matrix, triangle);
		requireStructure(canonical); float[] result = new float[columnIndices.length];
		for (int source = 0; source < canonical.keys.length; source++) {
			long key = canonical.keys[source]; int first = inverse[row(key)], second = inverse[column(key)];
			int factorRow = Math.max(first, second), factorColumn = Math.min(first, second);
			result[offset(factorRow, factorColumn)] = canonical.floatValues[source];
		}
		return result;
	}

	private int offset(int row, int column) {
		int found = Arrays.binarySearch(columnIndices, rowStarts[row], rowStarts[row + 1], column);
		if (found < 0) throw new IllegalStateException("input entry is absent from sparse factor pattern");
		return found;
	}
	private void requireStructure(Canonical canonical) {
		if (canonical.dimension != dimension || !Arrays.equals(canonical.keys, structure))
			throw new IllegalArgumentException("sparse refactorization structure differs from analysis");
	}
	private static int minimumDegree(Set<Integer>[] graph, boolean[] active) {
		int selected = -1, degree = Integer.MAX_VALUE;
		for (int candidate = 0; candidate < active.length; candidate++) if (active[candidate]) {
			int value = 0; for (int neighbor : graph[candidate]) if (active[neighbor]) value++;
			if (value < degree) { selected = candidate; degree = value; }
		}
		return selected;
	}
	private static long key(int row, int column) { return ((long) row << 32) | (column & 0xffffffffL); }
	private static int row(long key) { return (int) (key >>> 32); }
	private static int column(long key) { return (int) key; }

	private static final class Canonical {
		final int dimension; final long[] keys; final double[] doubleValues; final float[] floatValues;
		Canonical(int dimension, long[] keys, double[] doubleValues, float[] floatValues) {
			this.dimension = dimension; this.keys = keys; this.doubleValues = doubleValues;
			this.floatValues = floatValues;
		}
		static Canonical create(CsrMatrix matrix, MatrixTriangle triangle) {
			if (matrix == null || triangle == null || matrix.rows() < 1 || matrix.rows() != matrix.columns())
				throw new IllegalArgumentException("sparse Cholesky requires a square matrix and triangle");
			Map<Long, Double> values = new HashMap<Long, Double>(); double[] source = matrix.values();
			int[] columns = matrix.columnIndices(), starts = matrix.rowStarts();
			for (int row = 0; row < matrix.rows(); row++) for (int at = starts[row] - 1; at < starts[row + 1] - 1; at++) {
				int column = columns[at] - 1; double value = source[at];
				if (!Double.isFinite(value)) throw new IllegalArgumentException("sparse matrix must be finite");
				if ((triangle == MatrixTriangle.LOWER && column <= row) || (triangle == MatrixTriangle.UPPER && column >= row)) {
					long key = key(Math.max(row, column), Math.min(row, column)); Double previous = values.get(key);
					values.put(key, previous == null ? value : previous + value);
				}
			}
			long[] keys = sortedKeys(values.keySet()); double[] result = new double[keys.length];
			for (int i = 0; i < keys.length; i++) result[i] = values.get(keys[i]);
			return new Canonical(matrix.rows(), keys, result, null);
		}
		static Canonical create(FloatCsrMatrix matrix, MatrixTriangle triangle) {
			if (matrix == null || triangle == null || matrix.rows() < 1 || matrix.rows() != matrix.columns())
				throw new IllegalArgumentException("FP32 sparse Cholesky requires a square matrix and triangle");
			Map<Long, Float> values = new HashMap<Long, Float>(); float[] source = matrix.values();
			int[] columns = matrix.columnIndices(), starts = matrix.rowStarts();
			for (int row = 0; row < matrix.rows(); row++) for (int at = starts[row] - 1; at < starts[row + 1] - 1; at++) {
				int column = columns[at] - 1; float value = source[at];
				if (!Float.isFinite(value)) throw new IllegalArgumentException("FP32 sparse matrix must be finite");
				if ((triangle == MatrixTriangle.LOWER && column <= row) || (triangle == MatrixTriangle.UPPER && column >= row)) {
					long key = key(Math.max(row, column), Math.min(row, column)); Float previous = values.get(key);
					values.put(key, previous == null ? value : previous + value);
				}
			}
			long[] keys = sortedKeys(values.keySet()); float[] result = new float[keys.length];
			for (int i = 0; i < keys.length; i++) result[i] = values.get(keys[i]);
			return new Canonical(matrix.rows(), keys, null, result);
		}
		private static long[] sortedKeys(Set<Long> input) {
			Long[] boxed = input.toArray(new Long[input.size()]); Arrays.sort(boxed);
			long[] result = new long[boxed.length]; for (int i = 0; i < boxed.length; i++) result[i] = boxed[i];
			return result;
		}
	}
}
