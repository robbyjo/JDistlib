/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Reusable FP32 CSR handle; capabilities report whether storage is provider-resident. */
public interface PreparedFloatCsrMatrix extends AutoCloseable {
	int rows();
	int columns();
	int nonzeroCount();
	void multiply(float alpha, float[] x, float beta, float[] y);
	void multiply(float alpha, float[] right, int rightColumns, float beta,
			float[] result);
	@Override void close();
}
