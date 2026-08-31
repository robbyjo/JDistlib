/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP64 Cholesky factor that can solve multiple right sides in place. */
public interface PreparedCholesky extends AutoCloseable {
	int dimension();
	double logDeterminant();
	/** Replaces a row-major dimension-by-columns right side with its solution. */
	void solveInPlace(double[] right, int columns);
	@Override void close();
}
