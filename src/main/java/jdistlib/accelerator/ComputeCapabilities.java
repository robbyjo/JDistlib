/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable compute-device capabilities relevant to statistical kernels. */
public final class ComputeCapabilities {
	private final String backend, device;
	private final boolean doublePrecision, runtimeCompilation;
	private final boolean denseLinearAlgebra, sparseLinearAlgebra, nativeFactorizations;
	private final long globalMemoryBytes;
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes) {
		this(backend, device, doublePrecision, runtimeCompilation, globalMemoryBytes,
				false, false, false);
	}
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes, boolean denseLinearAlgebra,
			boolean sparseLinearAlgebra, boolean nativeFactorizations) {
		if (backend == null || device == null || globalMemoryBytes < 0L)
			throw new IllegalArgumentException("invalid compute capabilities");
		this.backend = backend; this.device = device;
		this.doublePrecision = doublePrecision;
		this.runtimeCompilation = runtimeCompilation;
		this.denseLinearAlgebra = denseLinearAlgebra;
		this.sparseLinearAlgebra = sparseLinearAlgebra;
		this.nativeFactorizations = nativeFactorizations;
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
	public long globalMemoryBytes() { return globalMemoryBytes; }
}
