/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** A row-by-feature matrix retained for repeated {@code X'v} score batches. */
public interface PreparedTransposeProduct extends AutoCloseable {
	int rows();
	int columns();
	/**
	 * Multiplies the transpose of the prepared matrix by one or more row vectors.
	 * Each input row must have {@link #rows()} values; each result row has
	 * {@link #columns()} values.
	 */
	double[][] multiply(double[][] vectors);
	@Override void close();
}
