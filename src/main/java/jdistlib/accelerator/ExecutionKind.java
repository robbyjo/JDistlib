/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** How an inspected operation is expected to execute. */
public enum ExecutionKind {
	JAVA_REFERENCE, NATIVE_CPU, GPU_PARALLEL, GPU_SERIAL, PORTABLE_FALLBACK, AUTOMATIC
}
