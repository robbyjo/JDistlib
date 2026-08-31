/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Selected compute backend, device provenance, and ownership for one workflow. */
public final class ComputeSelection implements AutoCloseable {
	private final Compute requested;
	private final ComputeBackend backend;
	private final String selectedBackend;
	private final boolean automaticRouting;
	ComputeSelection(Compute requested, ComputeBackend backend,
			String selectedBackend, boolean automaticRouting) {
		this.requested = requested; this.backend = backend;
		this.selectedBackend = selectedBackend; this.automaticRouting = automaticRouting;
	}
	public Compute requested() { return requested; }
	/** Returns the owned backend; closing this selection closes it. */
	public ComputeBackend backend() { return backend; }
	/** Returns the concrete provider used for accelerated work, or {@code cpu}. */
	public String selectedBackend() { return selectedBackend; }
	public String device() { return backend.capabilities().device(); }
	/** Returns detailed runtime, driver, API, and device identification. */
	public ComputeDeviceInfo deviceInfo() { return backend.deviceInfo(); }
	/** Predicts the concrete execution route for an operation without running it. */
	public ExecutionPlan plan(LinearAlgebraOperation operation, NumericPrecision precision,
			int... dimensions) { return backend.plan(operation, precision, dimensions); }
	public boolean accelerated() { return !"cpu".equals(selectedBackend); }
	public boolean automaticRouting() { return automaticRouting; }
	public String description() {
		return "policy=" + requested.name().toLowerCase(java.util.Locale.ROOT)
				+ ", backend=" + selectedBackend + ", device=" + device()
				+ (automaticRouting ? ", thresholded CPU fallback enabled" : "");
	}
	@Override public void close() { backend.close(); }
}
