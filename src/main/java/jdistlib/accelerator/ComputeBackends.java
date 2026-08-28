/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;

/** Discovers optional accelerator providers without making them core dependencies. */
public final class ComputeBackends {
	private ComputeBackends() {}
	public static List<ComputeBackend> available() {
		List<ComputeBackend> result = new ArrayList<ComputeBackend>();
		result.add(new CpuComputeBackend());
		try {
			for (ComputeBackend backend : ServiceLoader.load(ComputeBackend.class))
				if (backend.available()) result.add(backend); else backend.close();
		} catch (ServiceConfigurationError error) {
			// A broken optional provider must never make the CPU core unusable.
		}
		return Collections.unmodifiableList(result);
	}
	public static ComputeBackend preferred() {
		List<ComputeBackend> candidates = available();
		String requested = System.getProperty("jdistlib.compute.backend", "auto").trim().toLowerCase(java.util.Locale.ROOT);
		if (!"auto".equals(requested)) {
			for (ComputeBackend backend : candidates) if (requested.equals(backend.id())) return retain(backend, candidates);
			close(candidates); throw new IllegalStateException("requested compute backend is unavailable: " + requested);
		}
		for (String id : new String[] {"cuda", "opencl", "cpu"})
			for (ComputeBackend backend : candidates) if (id.equals(backend.id())) return retain(backend, candidates);
		throw new AssertionError("CPU backend missing");
	}
	public static ComputeBackend byId(String id) {
		if (id == null) throw new IllegalArgumentException("backend id is required"); List<ComputeBackend> candidates = available();
		for (ComputeBackend backend : candidates) if (id.equalsIgnoreCase(backend.id())) return retain(backend, candidates);
		close(candidates); throw new IllegalStateException("compute backend is unavailable: " + id);
	}
	private static void close(List<ComputeBackend> backends) { for (ComputeBackend backend : backends) backend.close(); }
	private static ComputeBackend retain(ComputeBackend selected, List<ComputeBackend> backends) {
		for (ComputeBackend backend : backends) if (backend != selected) backend.close(); return selected;
	}
}
