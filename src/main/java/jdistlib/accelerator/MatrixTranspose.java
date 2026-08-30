/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Transposition applied to a row-major dense matrix operand. */
public enum MatrixTranspose {
	/** Use the stored matrix without transposition. */
	NONE,
	/** Use the transpose of the stored matrix. */
	TRANSPOSE
}
