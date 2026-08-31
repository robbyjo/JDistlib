/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.NativeLibrary;
import java.io.File;
import java.util.List;
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.accelerator.SparseOrdering;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Optional system Intel oneMKL CBLAS/LAPACKE provider. */
public final class OneMklComputeBackend extends NativeCpuComputeBackend {
	@Override public String id() { return "onemkl"; }
	@Override String propertyName() { return "jdistlib.onemkl.library"; }
	@Override String[] libraryNames() { return new String[] {"mkl_rt", "mkl_rt.2"}; }
	@Override String displayName() { return "oneMKL"; }
	@Override ComputeApi api() { return ComputeApi.ONEMKL; }
	@Override boolean nativeSparseFactorizations() {
		return optional("pardiso") != null && optional("pardisoinit") != null
				&& optional("pardiso_getdiag") != null;
	}
	@Override public PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		if (!nativeSparseFactorizations() || ordering == SparseOrdering.NATURAL)
			return super.prepareDcsrpotrf(matrix, triangle, ordering);
		return MklPardisoSolver.prepare(this, matrix, triangle);
	}
	@Override public PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		if (!nativeSparseFactorizations() || ordering == SparseOrdering.NATURAL)
			return super.prepareScsrpotrf(matrix, triangle, ordering);
		return MklPardisoSolver.prepare(this, matrix, triangle);
	}
	@Override List<File> installedLibraries() { return oneApiLibraries(); }
	@Override String detectRuntimeVersion(NativeLibrary loaded) {
		Function function = optional("MKL_Get_Version_String");
		if (function == null) return "unknown";
		byte[] value = new byte[256]; function.invokeVoid(new Object[] {value, value.length});
		int length = 0; while (length < value.length && value[length] != 0) length++;
		return new String(value, 0, length, java.nio.charset.StandardCharsets.US_ASCII).trim();
	}
}
