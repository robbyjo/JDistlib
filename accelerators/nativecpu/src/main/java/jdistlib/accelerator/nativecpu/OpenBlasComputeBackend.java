/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.NativeLibrary;
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.accelerator.SparseOrdering;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Optional system OpenBLAS CBLAS/LAPACKE provider. */
public final class OpenBlasComputeBackend extends NativeCpuComputeBackend {
	private NativeLibrary cholmod;
	private int cholmodMajor;
	private Throwable cholmodUnavailableCause;
	public OpenBlasComputeBackend() {
		super();
		if (available()) try {
			String configured = System.getProperty("jdistlib.cholmod.library");
			String[] candidates = configured == null || configured.trim().isEmpty()
					? new String[] {"cholmod", "libcholmod"}
					: new String[] {configured.trim()};
			Throwable last = null;
			for (String candidate : candidates) try { cholmod = NativeLibrary.getInstance(candidate); break; }
			catch (Throwable error) { last = error; }
			if (cholmod == null) cholmodUnavailableCause = last == null
					? new IllegalStateException("SuiteSparse CHOLMOD runtime is unavailable") : last;
			if (cholmod != null) {
				int[] version = new int[3]; cholmod.getFunction("cholmod_version").invokeInt(new Object[] {version});
				cholmodMajor = version[0];
				if (cholmodMajor < 5) { cholmodUnavailableCause = new UnsupportedOperationException(
						"SuiteSparse CHOLMOD 5 or later is required"); cholmod.close(); cholmod = null; }
				else for (String symbol : new String[] {"cholmod_start", "cholmod_finish",
						"cholmod_analyze", "cholmod_factorize", "cholmod_change_factor",
						"cholmod_copy_factor", "cholmod_factor_to_sparse", "cholmod_solve",
						"cholmod_free_sparse", "cholmod_free_factor", "cholmod_free_dense"})
					cholmod.getFunction(symbol);
			}
		} catch (Throwable unavailable) { cholmodUnavailableCause = unavailable;
			if (cholmod != null) cholmod.close(); cholmod = null; }
	}
	@Override public String id() { return "openblas"; }
	@Override String propertyName() { return "jdistlib.openblas.library"; }
	@Override String[] libraryNames() {
		return new String[] {"openblas", "libopenblas", "libopenblas64_"};
	}
	@Override String displayName() { return "OpenBLAS"; }
	@Override ComputeApi api() { return ComputeApi.OPENBLAS; }
	@Override boolean nativeSparseFactorizations() { return cholmod != null; }
	/** Returns why the optional CHOLMOD sparse companion could not be loaded. */
	public Throwable cholmodUnavailableCause() { return cholmodUnavailableCause; }
	Function cholmod(String name) {
		if (cholmod == null) throw new UnsupportedOperationException("SuiteSparse CHOLMOD 5 or later is unavailable");
		return cholmod.getFunction(name);
	}
	@Override public PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		if (!nativeSparseFactorizations() || ordering == SparseOrdering.NATURAL)
			return super.prepareDcsrpotrf(matrix, triangle, ordering);
		return CholmodSparseSolver.prepare(this, matrix, triangle);
	}
	@Override public PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		if (!nativeSparseFactorizations() || cholmodMajor < 5 || ordering == SparseOrdering.NATURAL)
			return super.prepareScsrpotrf(matrix, triangle, ordering);
		return CholmodSparseSolver.prepare(this, matrix, triangle);
	}
	@Override String detectRuntimeVersion(NativeLibrary loaded) {
		return pointerString(optional("openblas_get_config"));
	}
	@Override public void close() {
		if (cholmod != null) { cholmod.close(); cholmod = null; }
		super.close();
	}
}
