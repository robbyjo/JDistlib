/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP64 CSR handle; capabilities report whether storage is provider-resident. */
public interface PreparedCsrMatrix extends AutoCloseable {
	int rows();
	int columns();
	int nonzeroCount();
	/** Performs {@code y := alpha*A*x + beta*y}. */
	void multiply(double alpha, double[] x, double beta, double[] y);
	/** Performs {@code C := alpha*A*B + beta*C} for row-major dense matrices. */
	void multiply(double alpha, double[] right, int rightColumns, double beta,
			double[] result);
	@Override void close();
}
