/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.Locale;

/** Selects automatic, CPU, or required accelerator execution. */
public enum Compute {
	/** Route eligible large operations to an available accelerator and small work to CPU. */
	AUTO,
	/** Always use the deterministic CPU reference implementation. */
	CPU,
	/** Require the optional Intel oneMKL native CPU provider. */
	ONEMKL,
	/** Require the optional OpenBLAS native CPU provider. */
	OPENBLAS,
	/** Require any available hardware accelerator. */
	GPU,
	/** Require the optional CUDA provider. */
	CUDA,
	/** Require the optional OpenCL provider. */
	OPENCL,
	/** Require the optional Vulkan compute provider. */
	VULKAN;

	/** Parses a case-insensitive command-line or system-property value. */
	public static Compute parse(String value) {
		if (value == null) throw new IllegalArgumentException("compute backend is required");
		try { return valueOf(value.trim().toUpperCase(Locale.ROOT)); }
		catch (IllegalArgumentException exception) {
			throw new IllegalArgumentException("unknown compute backend: " + value
					+ " (expected auto, cpu, onemkl, openblas, gpu, cuda, opencl, or vulkan)", exception);
		}
	}
}
