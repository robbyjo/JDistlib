/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.FloatCsrMatrix;

/** Symbolically analyzed FP32 sparse Cholesky handle with reusable numeric factors. */
public interface PreparedFloatSparseCholesky extends AutoCloseable {
	int dimension();
	int structuralNonzeroCount();
	int factorNonzeroCount();
	int[] permutation();
	float logDeterminant();
	void refactor(FloatCsrMatrix matrix);
	void solveInPlace(float[] right, int columns);
	default float[] solve(float[] right) {
		if (right == null || right.length != dimension())
			throw new IllegalArgumentException("right side length must equal factor dimension");
		float[] result = right.clone(); solveInPlace(result, 1); return result;
	}
	default float[] solve(float[] right, int columns) {
		if (columns < 1 || right == null || right.length != dimension() * columns)
			throw new IllegalArgumentException("invalid FP32 sparse Cholesky right side");
		float[] result = right.clone(); solveInPlace(result, columns); return result;
	}
	@Override void close();
}
