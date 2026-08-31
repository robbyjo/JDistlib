/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Fill-reducing ordering used before a sparse symmetric factorization. */
public enum SparseOrdering {
	/** Preserve the input row and column order. */
	NATURAL,
	/** Greedily eliminate the active vertex with minimum graph degree. */
	MINIMUM_DEGREE
}
