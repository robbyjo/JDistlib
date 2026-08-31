/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.NativeLibrary;
import jdistlib.accelerator.ComputeApi;

/** Optional system OpenBLAS CBLAS/LAPACKE provider. */
public final class OpenBlasComputeBackend extends NativeCpuComputeBackend {
	@Override public String id() { return "openblas"; }
	@Override String propertyName() { return "jdistlib.openblas.library"; }
	@Override String[] libraryNames() {
		return new String[] {"openblas", "libopenblas", "libopenblas64_"};
	}
	@Override String displayName() { return "OpenBLAS"; }
	@Override ComputeApi api() { return ComputeApi.OPENBLAS; }
	@Override String detectRuntimeVersion(NativeLibrary loaded) {
		return pointerString(optional("openblas_get_config"));
	}
}
