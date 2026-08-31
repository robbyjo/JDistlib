/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;

/** Symbolically analyzed FP64 sparse Cholesky handle with reusable numeric factors. */
public interface PreparedSparseCholesky extends AutoCloseable {
	int dimension();
	/** Number of unique entries in the authoritative input triangle. */
	int structuralNonzeroCount();
	/** Number of entries in the current lower Cholesky factor. */
	int factorNonzeroCount();
	/** Returns the new-to-original permutation, or an empty array when provider-owned. */
	int[] permutation();
	double logDeterminant();
	/** Replaces only numerical values; the authoritative-triangle structure must match. */
	void refactor(CsrMatrix matrix);
	/** Replaces a row-major dimension-by-columns right side with its solution. */
	void solveInPlace(double[] right, int columns);
	default double[] solve(double[] right) {
		if (right == null || right.length != dimension())
			throw new IllegalArgumentException("right side length must equal factor dimension");
		double[] result = right.clone(); solveInPlace(result, 1); return result;
	}
	default double[] solve(double[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension() * columns)
			throw new IllegalArgumentException("invalid sparse Cholesky right side");
		double[] result = right.clone(); solveInPlace(result, columns); return result;
	}
	@Override void close();
}
