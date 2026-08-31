/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Portable prepared sparse handles shared by default backend implementations. */
final class CpuPreparedSparse {
	private CpuPreparedSparse() {}
	static PreparedCsrMatrix matrix(final LinearAlgebraBackend backend,
			final CsrMatrix matrix) {
		if (matrix == null) throw new IllegalArgumentException("CSR matrix is required");
		return new PreparedCsrMatrix() {
			private boolean closed;
			@Override public int rows() { check(); return matrix.rows(); }
			@Override public int columns() { check(); return matrix.columns(); }
			@Override public int nonzeroCount() { check(); return matrix.nonzeroCount(); }
			@Override public void multiply(double alpha, double[] x, double beta, double[] y) {
				check(); backend.dcsrmv(alpha, matrix, x, beta, y);
			}
			@Override public void multiply(double alpha, double[] right, int rightColumns,
					double beta, double[] result) {
				check(); backend.dcsrmm(alpha, matrix, right, rightColumns, beta, result);
			}
			@Override public void close() { closed = true; }
			private void check() { if (closed) throw new IllegalStateException("prepared CSR matrix is closed"); }
		};
	}
	static PreparedFloatCsrMatrix matrix(final SinglePrecisionLinearAlgebraBackend backend,
			final FloatCsrMatrix matrix) {
		if (matrix == null) throw new IllegalArgumentException("FP32 CSR matrix is required");
		return new PreparedFloatCsrMatrix() {
			private boolean closed;
			@Override public int rows() { check(); return matrix.rows(); }
			@Override public int columns() { check(); return matrix.columns(); }
			@Override public int nonzeroCount() { check(); return matrix.nonzeroCount(); }
			@Override public void multiply(float alpha, float[] x, float beta, float[] y) {
				check(); backend.scsrmv(alpha, matrix, x, beta, y);
			}
			@Override public void multiply(float alpha, float[] right, int rightColumns,
					float beta, float[] result) {
				check(); backend.scsrmm(alpha, matrix, right,
						rightColumns, beta, result);
			}
			@Override public void close() { closed = true; }
			private void check() { if (closed) throw new IllegalStateException("prepared FP32 CSR matrix is closed"); }
		};
	}
}
