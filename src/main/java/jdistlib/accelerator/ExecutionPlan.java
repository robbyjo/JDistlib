/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable prediction of where and how one operation will execute. */
public final class ExecutionPlan {
	private final LinearAlgebraOperation operation; private final NumericPrecision precision;
	private final ExecutionKind kind; private final String backendId, device, reason;
	public ExecutionPlan(LinearAlgebraOperation operation, NumericPrecision precision,
			ExecutionKind kind, String backendId, String device, String reason) {
		if (operation == null || precision == null || kind == null || backendId == null
				|| device == null || reason == null)
			throw new IllegalArgumentException("execution-plan fields are required");
		this.operation = operation; this.precision = precision; this.kind = kind;
		this.backendId = backendId; this.device = device; this.reason = reason;
	}
	public LinearAlgebraOperation operation() { return operation; }
	public NumericPrecision precision() { return precision; }
	public ExecutionKind kind() { return kind; }
	public String backendId() { return backendId; }
	public String device() { return device; }
	public String reason() { return reason; }
	public boolean accelerated() {
		return kind == ExecutionKind.NATIVE_CPU || kind == ExecutionKind.GPU_PARALLEL
				|| kind == ExecutionKind.GPU_SERIAL;
	}
	public String description() {
		return operation + " " + precision + " -> " + backendId + " (" + kind + ", "
				+ reason + ")";
	}
}
