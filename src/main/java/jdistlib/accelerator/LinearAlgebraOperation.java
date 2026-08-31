/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Backend-neutral operation identifiers used for capability and routing inspection. */
public enum LinearAlgebraOperation {
	AXPY, DOT, NRM2, GEMV, GEMM, SYRK, TRSV, TRSM,
	CSR_MV, CSR_MM, PREPARED_CSR, CSR_POTRF, CSR_ANALYZE, CSR_REFACTOR, CSR_SOLVE,
	POTRF, GEQP3, SYEV, GESVD, PREPARED_CHOLESKY
}
