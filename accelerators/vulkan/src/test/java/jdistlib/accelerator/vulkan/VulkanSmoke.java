/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.vulkan;

import java.util.Arrays;

/** Explicit hardware smoke for maintainers; unlike unit tests, absence is an error. */
public final class VulkanSmoke {
	private VulkanSmoke() {}
	public static void main(String[] arguments) {
		VulkanComputeBackend backend = new VulkanComputeBackend();
		try {
			if (!backend.available()) throw new IllegalStateException("Vulkan unavailable", backend.unavailableCause());
			double[] result = backend.axpy(2.0, new double[] {1.0, 2.0}, new double[] {3.0, 4.0});
			if (!Arrays.equals(result, new double[] {5.0, 8.0})) throw new AssertionError(Arrays.toString(result));
			System.out.println("Vulkan smoke passed on " + backend.capabilities().device());
		} finally { backend.close(); }
	}
}
