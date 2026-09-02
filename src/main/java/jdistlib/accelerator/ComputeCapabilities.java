/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable compute-device capabilities relevant to statistical kernels. */
public final class ComputeCapabilities {
	private final String backend, device;
	private final boolean doublePrecision, runtimeCompilation;
	private final boolean denseLinearAlgebra, sparseLinearAlgebra, nativeFactorizations;
	private final boolean preparedSparseMatrices, nativeSparseFactorizations;
	private final boolean reusableSparseFactorizations;
	private final boolean preparedDenseMatrices, batchedLinearAlgebra;
	private final long globalMemoryBytes;
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes) {
		this(backend, device, doublePrecision, runtimeCompilation, globalMemoryBytes,
				false, false, false);
	}
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes, boolean denseLinearAlgebra,
			boolean sparseLinearAlgebra, boolean nativeFactorizations) {
		this(backend, device, doublePrecision, runtimeCompilation, globalMemoryBytes,
				denseLinearAlgebra, sparseLinearAlgebra, nativeFactorizations,
				false, false, true);
	}
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes, boolean denseLinearAlgebra,
			boolean sparseLinearAlgebra, boolean nativeFactorizations,
			boolean preparedSparseMatrices, boolean nativeSparseFactorizations,
			boolean reusableSparseFactorizations) {
		this(backend, device, doublePrecision, runtimeCompilation, globalMemoryBytes,
				denseLinearAlgebra, sparseLinearAlgebra, nativeFactorizations,
				preparedSparseMatrices, nativeSparseFactorizations,
				reusableSparseFactorizations, false, false);
	}
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes, boolean denseLinearAlgebra,
			boolean sparseLinearAlgebra, boolean nativeFactorizations,
			boolean preparedSparseMatrices, boolean nativeSparseFactorizations,
			boolean reusableSparseFactorizations, boolean preparedDenseMatrices,
			boolean batchedLinearAlgebra) {
		if (backend == null || device == null || globalMemoryBytes < 0L)
			throw new IllegalArgumentException("invalid compute capabilities");
		this.backend = backend; this.device = device;
		this.doublePrecision = doublePrecision;
		this.runtimeCompilation = runtimeCompilation;
		this.denseLinearAlgebra = denseLinearAlgebra;
		this.sparseLinearAlgebra = sparseLinearAlgebra;
		this.nativeFactorizations = nativeFactorizations;
		this.preparedSparseMatrices = preparedSparseMatrices;
		this.nativeSparseFactorizations = nativeSparseFactorizations;
		this.reusableSparseFactorizations = reusableSparseFactorizations;
		this.preparedDenseMatrices = preparedDenseMatrices;
		this.batchedLinearAlgebra = batchedLinearAlgebra;
		this.globalMemoryBytes = globalMemoryBytes;
	}
	public String backend() { return backend; }
	public String device() { return device; }
	public boolean doublePrecision() { return doublePrecision; }
	public boolean runtimeCompilation() { return runtimeCompilation; }
	/** Whether the provider accelerates the public dense BLAS operations. */
	public boolean denseLinearAlgebra() { return denseLinearAlgebra; }
	/** Whether the provider accelerates the public CSR operations. */
	public boolean sparseLinearAlgebra() { return sparseLinearAlgebra; }
	/** Whether factorization methods execute natively on this provider. */
	public boolean nativeFactorizations() { return nativeFactorizations; }
	/** Whether prepared CSR handles retain provider-optimal storage between calls. */
	public boolean preparedSparseMatrices() { return preparedSparseMatrices; }
	/** Whether sparse Cholesky numeric factorization and solves execute on the provider. */
	public boolean nativeSparseFactorizations() { return nativeSparseFactorizations; }
	/** Whether sparse symbolic analysis is reusable when only numerical values change. */
	public boolean reusableSparseFactorizations() { return reusableSparseFactorizations; }
	/** Whether prepared dense handles retain provider-optimal storage between calls. */
	public boolean preparedDenseMatrices() { return preparedDenseMatrices; }
	/** Whether the provider accepts batched dense operations through the public contract. */
	public boolean batchedLinearAlgebra() { return batchedLinearAlgebra; }
	public long globalMemoryBytes() { return globalMemoryBytes; }
}
