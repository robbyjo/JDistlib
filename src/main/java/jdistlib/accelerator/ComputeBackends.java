/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.logging.Logger;

/** Discovers optional accelerator providers without making them core dependencies. */
public final class ComputeBackends {
	private static final Logger LOG = Logger.getLogger(ComputeBackends.class.getName());
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
		String configured = System.getProperty("jdistlib.compute.backend", "auto");
		try { return select(Compute.parse(configured)).backend(); }
		catch (IllegalArgumentException customProvider) { return byId(configured); }
	}
	/** Selects and reports an owned backend according to the requested policy. */
	public static ComputeSelection select(Compute requested) {
		if (requested == null) throw new IllegalArgumentException("compute policy is required");
		List<ComputeBackend> candidates = new ArrayList<ComputeBackend>(available());
		ComputeBackend cpu = find("cpu", candidates);
		if (requested == Compute.CPU) return report(new ComputeSelection(requested,
				retain(cpu, candidates), "cpu", false));
		if (requested == Compute.AUTO) {
			ComputeBackend accelerator = firstAccelerator(candidates);
			if (accelerator == null) return report(new ComputeSelection(requested,
					retain(cpu, candidates), "cpu", false));
			remove(cpu, candidates); remove(accelerator, candidates); close(candidates);
			return report(new ComputeSelection(requested,
					new AutoComputeBackend(cpu, accelerator), accelerator.id(), true));
		}
		ComputeBackend selected = requested == Compute.GPU
				? firstAccelerator(candidates)
				: find(requested.name().toLowerCase(java.util.Locale.ROOT), candidates);
		if (selected == null || "cpu".equals(selected.id())) {
			close(candidates);
			throw new IllegalStateException("requested compute backend is unavailable: "
					+ requested.name().toLowerCase(java.util.Locale.ROOT));
		}
		String selectedId = selected.id();
		return report(new ComputeSelection(requested, retain(selected, candidates),
				selectedId, false));
	}
	public static ComputeBackend byId(String id) {
		if (id == null) throw new IllegalArgumentException("backend id is required");
		try { return select(Compute.parse(id)).backend(); }
		catch (IllegalArgumentException unknownPolicy) {
			List<ComputeBackend> candidates = new ArrayList<ComputeBackend>(available());
			for (ComputeBackend backend : candidates)
				if (id.equalsIgnoreCase(backend.id())) return retain(backend, candidates);
			close(candidates); throw new IllegalStateException(
					"compute backend is unavailable: " + id, unknownPolicy);
		}
	}
	private static ComputeSelection report(ComputeSelection selection) {
		LOG.info("JDistlib compute: " + selection.description()); return selection;
	}
	private static ComputeBackend firstAccelerator(List<ComputeBackend> candidates) {
		for (String id : new String[] {"cuda", "opencl", "vulkan"}) {
			ComputeBackend value = find(id, candidates); if (value != null) return value;
		}
		return null;
	}
	private static ComputeBackend find(String id, List<ComputeBackend> candidates) {
		for (ComputeBackend backend : candidates) if (id.equals(backend.id())) return backend;
		return null;
	}
	private static void remove(ComputeBackend backend, List<ComputeBackend> backends) {
		backends.remove(backend);
	}
	private static void close(List<ComputeBackend> backends) { for (ComputeBackend backend : backends) backend.close(); }
	private static ComputeBackend retain(ComputeBackend selected, List<ComputeBackend> backends) {
		for (ComputeBackend backend : backends) if (backend != selected) backend.close(); return selected;
	}
}
