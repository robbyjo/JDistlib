/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Backend-neutral operation identifiers used for capability and routing inspection. */
public enum LinearAlgebraOperation {
	SCAL, COPY, SWAP, ASUM, IAMAX, AXPY, DOT, NRM2,
	GEMV, GEMM, BATCHED_GEMM, GER, SYR, SYR2, SYRK, SYR2K, SYMM, TRSV, TRSM,
	CSR_MV, CSR_MM, CSR_GEMM, CSR_TRSV, PREPARED_CSR, PREPARED_DENSE,
	CSR_POTRF, CSR_ANALYZE, CSR_REFACTOR, CSR_SOLVE,
	POTRF, BATCHED_POTRF, GETRF, BATCHED_GETRF, SYTRF, GEQP3, SYEV, SYGVD,
	GESVD, PREPARED_CHOLESKY
}
