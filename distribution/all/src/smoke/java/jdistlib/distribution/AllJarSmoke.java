/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.distribution;

import java.util.List;

import jdistlib.Normal;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeBackends;

/** Runtime smoke executed with only jdistlib-all on the application class path. */
public final class AllJarSmoke {
	private AllJarSmoke() {}
	public static void main(String[] arguments) {
		double density = Normal.density(0.0, 0.0, 1.0, false);
		if (!(density > 0.398 && density < 0.399)) throw new AssertionError("core distribution unavailable");
		List<ComputeBackend> backends = ComputeBackends.available();
		boolean cpu = false;
		try {
			for (ComputeBackend backend : backends) {
				if ("cpu".equals(backend.id())) cpu = true;
				double[] value = backend.axpy(2.0, new double[] {1.0, 2.0}, new double[] {3.0, 4.0});
				if (value.length != 2 || value[0] != 5.0 || value[1] != 8.0)
					throw new AssertionError("backend failed from unified JAR: " + backend.id());
			}
			if (!cpu) throw new AssertionError("CPU provider unavailable");
		} finally {
			for (ComputeBackend backend : backends) backend.close();
		}
		System.out.println("JDistlib all-in-one smoke passed; detected " + backends.size() + " backend(s)");
	}
}
