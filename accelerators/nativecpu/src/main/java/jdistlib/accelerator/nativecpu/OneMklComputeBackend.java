/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.NativeLibrary;
import java.io.File;
import java.util.List;
import jdistlib.accelerator.ComputeApi;

/** Optional system Intel oneMKL CBLAS/LAPACKE provider. */
public final class OneMklComputeBackend extends NativeCpuComputeBackend {
	@Override public String id() { return "onemkl"; }
	@Override String propertyName() { return "jdistlib.onemkl.library"; }
	@Override String[] libraryNames() { return new String[] {"mkl_rt", "mkl_rt.2"}; }
	@Override String displayName() { return "oneMKL"; }
	@Override ComputeApi api() { return ComputeApi.ONEMKL; }
	@Override List<File> installedLibraries() { return oneApiLibraries(); }
	@Override String detectRuntimeVersion(NativeLibrary loaded) {
		Function function = optional("MKL_Get_Version_String");
		if (function == null) return "unknown";
		byte[] value = new byte[256]; function.invokeVoid(new Object[] {value, value.length});
		int length = 0; while (length < value.length && value[length] != 0) length++;
		return new String(value, 0, length, java.nio.charset.StandardCharsets.US_ASCII).trim();
	}
}
