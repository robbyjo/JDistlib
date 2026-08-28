/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Locale;

/** Controls whether NUTS target evaluation may or must use an accelerator. */
public enum ComputeNuts {
	/** Require CPU target evaluation for NUTS. */
	OFF,
	/** Use the configured compute policy and its profitability thresholds. */
	AUTO,
	/** Require hardware-accelerated target evaluation even when it may be slower. */
	FORCE;

	/** Parses a case-insensitive command-line or system-property value. */
	public static ComputeNuts parse(String value) {
		if (value == null) throw new IllegalArgumentException("NUTS offload policy is required");
		try { return valueOf(value.trim().toUpperCase(Locale.ROOT)); }
		catch (IllegalArgumentException exception) {
			throw new IllegalArgumentException("unknown NUTS offload policy: " + value
					+ " (expected off, auto, or force)", exception);
		}
	}
}
