/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP32 Cholesky factor that can solve multiple right sides in place. */
public interface PreparedFloatCholesky extends AutoCloseable {
	int dimension();
	float logDeterminant();
	/** Replaces a row-major dimension-by-columns right side with its solution. */
	void solveInPlace(float[] right, int columns);
	@Override void close();
}
