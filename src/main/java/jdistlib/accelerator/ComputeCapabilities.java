/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable compute-device capabilities relevant to statistical kernels. */
public final class ComputeCapabilities {
	private final String backend, device;
	private final boolean doublePrecision, runtimeCompilation;
	private final long globalMemoryBytes;
	public ComputeCapabilities(String backend, String device, boolean doublePrecision,
			boolean runtimeCompilation, long globalMemoryBytes) {
		if (backend == null || device == null || globalMemoryBytes < 0L)
			throw new IllegalArgumentException("invalid compute capabilities");
		this.backend = backend; this.device = device;
		this.doublePrecision = doublePrecision;
		this.runtimeCompilation = runtimeCompilation;
		this.globalMemoryBytes = globalMemoryBytes;
	}
	public String backend() { return backend; }
	public String device() { return device; }
	public boolean doublePrecision() { return doublePrecision; }
	public boolean runtimeCompilation() { return runtimeCompilation; }
	public long globalMemoryBytes() { return globalMemoryBytes; }
}
