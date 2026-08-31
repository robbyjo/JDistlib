/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import jdistlib.accelerator.CholeskyFactor;
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeCapabilities;
import jdistlib.accelerator.ComputeDeviceInfo;
import jdistlib.accelerator.FloatCholeskyFactor;
import jdistlib.accelerator.FloatPivotedQrFactor;
import jdistlib.accelerator.FloatSingularValueDecomposition;
import jdistlib.accelerator.FloatSymmetricEigenDecomposition;
import jdistlib.accelerator.MatrixDiagonal;
import jdistlib.accelerator.MatrixSide;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PivotedQrFactor;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SparseOrdering;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Shared dynamic CBLAS/LAPACKE adapter. Native libraries remain system dependencies. */
abstract class NativeCpuComputeBackend implements ComputeBackend {
	private static final int ROW_MAJOR = 101;
	private static final int NO_TRANS = 111, TRANS = 112;
	private static final int UPPER = 121, LOWER = 122;
	private static final int NON_UNIT = 131, UNIT = 132;
	private static final int LEFT = 141, RIGHT = 142;
	private NativeLibrary library;
	private Throwable unavailableCause;
	private String runtimeVersion = "unknown";
	private boolean lapacke;

	NativeCpuComputeBackend() {
		try {
			library = loadLibrary();
			runtimeVersion = detectRuntimeVersion(library);
			lapacke = optional("LAPACKE_dpotrf") != null;
			function("cblas_dgemm");
		} catch (Throwable error) {
			unavailableCause = error;
			close();
		}
	}

	abstract String propertyName();
	abstract String[] libraryNames();
	abstract String displayName();
	abstract ComputeApi api();
	abstract String detectRuntimeVersion(NativeLibrary loaded);
	List<File> installedLibraries() { return Collections.emptyList(); }
	boolean nativeSparseFactorizations() { return false; }
	boolean preparedSparseMatrices() { return false; }

	@Override public final boolean available() { return unavailableCause == null && library != null; }
	/** Returns the dynamic-loader failure when this optional provider is unavailable. */
	public final Throwable unavailableCause() { return unavailableCause; }
	@Override public final ComputeCapabilities capabilities() {
		ensureAvailable();
		return new ComputeCapabilities(displayName(), System.getProperty("os.arch", "unknown"),
				true, false, 0L, true, false, lapacke, preparedSparseMatrices(),
				nativeSparseFactorizations(), true);
	}
	@Override public final ComputeDeviceInfo deviceInfo() {
		ensureAvailable();
		Package pkg = getClass().getPackage();
		String providerVersion = pkg == null || pkg.getImplementationVersion() == null
				? "development" : pkg.getImplementationVersion();
		return new ComputeDeviceInfo(id(), providerVersion, api(), runtimeVersion,
				"n/a", displayName(), System.getProperty("os.arch", "unknown"),
				System.getProperty("os.arch", "unknown"), "host", 0L);
	}

	@Override public final void daxpy(int count, double alpha, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (xOffset != 0 || yOffset != 0) {
			ComputeBackend.super.daxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
			return;
		}
		function("cblas_daxpy").invokeVoid(new Object[] {count, alpha, x, xStride, y, yStride});
	}
	@Override public final double ddot(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (xOffset != 0 || yOffset != 0) return ComputeBackend.super.ddot(count, x, xOffset,
				xStride, y, yOffset, yStride);
		return function("cblas_ddot").invokeDouble(new Object[] {count, x, xStride, y, yStride});
	}
	@Override public final double dnrm2(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride);
		if (offset != 0) return ComputeBackend.super.dnrm2(count, x, offset, stride);
		return function("cblas_dnrm2").invokeDouble(new Object[] {count, x, stride});
	}
	@Override public final void dgemv(MatrixTranspose transpose, int rows, int columns,
			double alpha, double[] matrix, double[] x, double beta, double[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y);
		function("cblas_dgemv").invokeVoid(new Object[] {ROW_MAJOR, trans(transpose), rows,
				columns, alpha, matrix, columns, x, 1, beta, y, 1});
	}
	@Override public final void dgemm(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared, double alpha,
			double[] left, double[] right, double beta, double[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result);
		int lda = leftTranspose == MatrixTranspose.NONE ? shared : rows;
		int ldb = rightTranspose == MatrixTranspose.NONE ? columns : shared;
		function("cblas_dgemm").invokeVoid(new Object[] {ROW_MAJOR, trans(leftTranspose),
				trans(rightTranspose), rows, columns, shared, alpha, left, lda, right, ldb,
				beta, result, columns});
	}
	@Override public final void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, double beta, double[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result);
		int lda = transpose == MatrixTranspose.NONE ? shared : dimension;
		function("cblas_dsyrk").invokeVoid(new Object[] {ROW_MAJOR, LOWER, trans(transpose),
				dimension, shared, alpha, matrix, lda, beta, result, dimension});
		mirrorLower(result, dimension);
	}
	@Override public final void dtrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector, dimension);
		function("cblas_dtrsv").invokeVoid(new Object[] {ROW_MAJOR, triangle(triangle),
				trans(transpose), diagonal(diagonal), dimension, matrix, dimension, vector, 1});
	}
	@Override public final void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, double[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right);
		int order = side == MatrixSide.LEFT ? rows : columns;
		function("cblas_dtrsm").invokeVoid(new Object[] {ROW_MAJOR, side(side),
				triangle(triangle), trans(transpose), diagonal(diagonal), rows, columns,
				alpha, matrix, order, right, columns});
	}

	@Override public final void saxpy(int count, float alpha, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (xOffset != 0 || yOffset != 0) {
			ComputeBackend.super.saxpy(count, alpha, x, xOffset, xStride, y, yOffset, yStride);
			return;
		}
		function("cblas_saxpy").invokeVoid(new Object[] {count, alpha, x, xStride, y, yStride});
	}
	@Override public final float sdot(int count, float[] x, int xOffset, int xStride,
			float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (xOffset != 0 || yOffset != 0) return ComputeBackend.super.sdot(count, x, xOffset,
				xStride, y, yOffset, yStride);
		return function("cblas_sdot").invokeFloat(new Object[] {count, x, xStride, y, yStride});
	}
	@Override public final float snrm2(int count, float[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride);
		if (offset != 0) return ComputeBackend.super.snrm2(count, x, offset, stride);
		return function("cblas_snrm2").invokeFloat(new Object[] {count, x, stride});
	}
	@Override public final void sgemv(MatrixTranspose transpose, int rows, int columns,
			float alpha, float[] matrix, float[] x, float beta, float[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y);
		function("cblas_sgemv").invokeVoid(new Object[] {ROW_MAJOR, trans(transpose), rows,
				columns, alpha, matrix, columns, x, 1, beta, y, 1});
	}
	@Override public final void sgemm(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared, float alpha,
			float[] left, float[] right, float beta, float[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result);
		int lda = leftTranspose == MatrixTranspose.NONE ? shared : rows;
		int ldb = rightTranspose == MatrixTranspose.NONE ? columns : shared;
		function("cblas_sgemm").invokeVoid(new Object[] {ROW_MAJOR, trans(leftTranspose),
				trans(rightTranspose), rows, columns, shared, alpha, left, lda, right, ldb,
				beta, result, columns});
	}
	@Override public final void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, float beta, float[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result);
		int lda = transpose == MatrixTranspose.NONE ? shared : dimension;
		function("cblas_ssyrk").invokeVoid(new Object[] {ROW_MAJOR, LOWER, trans(transpose),
				dimension, shared, alpha, matrix, lda, beta, result, dimension});
		mirrorLower(result, dimension);
	}
	@Override public final void strsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector, dimension);
		function("cblas_strsv").invokeVoid(new Object[] {ROW_MAJOR, triangle(triangle),
				trans(transpose), diagonal(diagonal), dimension, matrix, dimension, vector, 1});
	}
	@Override public final void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, float[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right);
		int order = side == MatrixSide.LEFT ? rows : columns;
		function("cblas_strsm").invokeVoid(new Object[] {ROW_MAJOR, side(side),
				triangle(triangle), trans(transpose), diagonal(diagonal), rows, columns,
				alpha, matrix, order, right, columns});
	}

	@Override public final CholeskyFactor dpotrf(double[] matrix, int dimension) {
		if (!lapacke) return ComputeBackend.super.dpotrf(matrix, dimension);
		checkSquare(matrix, dimension); double[] factor = matrix.clone();
		int info = function("LAPACKE_dpotrf").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'L'), dimension, factor, dimension});
		checkInfo("dpotrf", info); clearUpper(factor, dimension);
		return new CholeskyFactor(dimension, factor);
	}
	@Override public final FloatCholeskyFactor spotrf(float[] matrix, int dimension) {
		if (!lapacke) return ComputeBackend.super.spotrf(matrix, dimension);
		checkSquare(matrix, dimension); float[] factor = matrix.clone();
		int info = function("LAPACKE_spotrf").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'L'), dimension, factor, dimension});
		checkInfo("spotrf", info); clearUpper(factor, dimension);
		return new FloatCholeskyFactor(dimension, factor);
	}
	@Override public final PivotedQrFactor dgeqp3(double[] matrix, int rows, int columns) {
		if (!lapacke) return ComputeBackend.super.dgeqp3(matrix, rows, columns);
		checkMatrix(matrix, rows, columns); double[] qr = matrix.clone();
		int[] pivot = new int[columns]; double[] tau = new double[Math.min(rows, columns)];
		int info = function("LAPACKE_dgeqp3").invokeInt(new Object[] {ROW_MAJOR, rows,
				columns, qr, columns, pivot, tau});
		checkInfo("dgeqp3", info); zeroBase(pivot);
		return new PivotedQrFactor(rows, columns, qr, tau, pivot);
	}
	@Override public final FloatPivotedQrFactor sgeqp3(float[] matrix, int rows, int columns) {
		if (!lapacke) return ComputeBackend.super.sgeqp3(matrix, rows, columns);
		checkMatrix(matrix, rows, columns); float[] qr = matrix.clone();
		int[] pivot = new int[columns]; float[] tau = new float[Math.min(rows, columns)];
		int info = function("LAPACKE_sgeqp3").invokeInt(new Object[] {ROW_MAJOR, rows,
				columns, qr, columns, pivot, tau});
		checkInfo("sgeqp3", info); zeroBase(pivot);
		return new FloatPivotedQrFactor(rows, columns, qr, tau, pivot);
	}
	@Override public final SymmetricEigenDecomposition dsyev(double[] matrix, int dimension) {
		if (!lapacke) return ComputeBackend.super.dsyev(matrix, dimension);
		checkSquare(matrix, dimension); double[] vectors = matrix.clone();
		double[] values = new double[dimension];
		int info = function("LAPACKE_dsyev").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'V'), Byte.valueOf((byte) 'L'), dimension, vectors,
				dimension, values});
		checkInfo("dsyev", info);
		return new SymmetricEigenDecomposition(dimension, values, vectors);
	}
	@Override public final FloatSymmetricEigenDecomposition ssyev(float[] matrix, int dimension) {
		if (!lapacke) return ComputeBackend.super.ssyev(matrix, dimension);
		checkSquare(matrix, dimension); float[] vectors = matrix.clone();
		float[] values = new float[dimension];
		int info = function("LAPACKE_ssyev").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'V'), Byte.valueOf((byte) 'L'), dimension, vectors,
				dimension, values});
		checkInfo("ssyev", info);
		return new FloatSymmetricEigenDecomposition(dimension, values, vectors);
	}
	@Override public final SingularValueDecomposition dgesvd(double[] matrix, int rows,
			int columns) {
		if (!lapacke) return ComputeBackend.super.dgesvd(matrix, rows, columns);
		checkMatrix(matrix, rows, columns); int count = Math.min(rows, columns);
		double[] work = matrix.clone(), values = new double[count];
		double[] left = new double[rows * count], right = new double[count * columns];
		double[] superb = new double[Math.max(1, count - 1)];
		int info = function("LAPACKE_dgesvd").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'S'), Byte.valueOf((byte) 'S'), rows, columns, work,
				columns, values, left, count, right, columns, superb});
		checkInfo("dgesvd", info);
		return new SingularValueDecomposition(rows, columns, values, left, right);
	}
	@Override public final FloatSingularValueDecomposition sgesvd(float[] matrix, int rows,
			int columns) {
		if (!lapacke) return ComputeBackend.super.sgesvd(matrix, rows, columns);
		checkMatrix(matrix, rows, columns); int count = Math.min(rows, columns);
		float[] work = matrix.clone(), values = new float[count];
		float[] left = new float[rows * count], right = new float[count * columns];
		float[] superb = new float[Math.max(1, count - 1)];
		int info = function("LAPACKE_sgesvd").invokeInt(new Object[] {ROW_MAJOR,
				Byte.valueOf((byte) 'S'), Byte.valueOf((byte) 'S'), rows, columns, work,
				columns, values, left, count, right, columns, superb});
		checkInfo("sgesvd", info);
		return new FloatSingularValueDecomposition(rows, columns, values, left, right);
	}
	@Override public PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return ComputeBackend.super.prepareDcsrpotrf(matrix, triangle, ordering);
	}
	@Override public PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return ComputeBackend.super.prepareScsrpotrf(matrix, triangle, ordering);
	}

	final Function optional(String name) {
		try { return library == null ? null : library.getFunction(name); }
		catch (UnsatisfiedLinkError error) { return null; }
	}
	final String pointerString(Function function) {
		if (function == null) return "unknown";
		Pointer value = (Pointer) function.invoke(Pointer.class, new Object[0]);
		return value == null ? "unknown" : value.getString(0L);
	}
	final Function function(String name) {
		ensureAvailable(); Function value = optional(name);
		if (value == null) throw new UnsupportedOperationException(displayName()
				+ " does not export " + name);
		return value;
	}
	private NativeLibrary loadLibrary() {
		List<String> candidates = new ArrayList<String>();
		String configured = System.getProperty(propertyName());
		if (configured != null && !configured.trim().isEmpty()) candidates.add(configured.trim());
		for (File installed : installedLibraries()) candidates.add(installed.getAbsolutePath());
		Collections.addAll(candidates, libraryNames());
		Throwable last = null;
		for (String candidate : candidates) try { return NativeLibrary.getInstance(candidate); }
		catch (Throwable error) { last = error; }
		throw new IllegalStateException("no loadable " + displayName() + " runtime found; set -D"
				+ propertyName() + "=<library path>", last);
	}
	@Override public final void close() {
		if (library != null) { library.close(); library = null; }
	}
	private void ensureAvailable() {
		if (!available()) throw new IllegalStateException(displayName() + " backend is unavailable",
				unavailableCause);
	}
	private static int trans(MatrixTranspose value) {
		if (value == null) throw new IllegalArgumentException("matrix transpose is required");
		return value == MatrixTranspose.NONE ? NO_TRANS : TRANS;
	}
	private static int triangle(MatrixTriangle value) {
		if (value == null) throw new IllegalArgumentException("matrix triangle is required");
		return value == MatrixTriangle.LOWER ? LOWER : UPPER;
	}
	private static int diagonal(MatrixDiagonal value) {
		if (value == null) throw new IllegalArgumentException("matrix diagonal is required");
		return value == MatrixDiagonal.UNIT ? UNIT : NON_UNIT;
	}
	private static int side(MatrixSide value) {
		if (value == null) throw new IllegalArgumentException("matrix side is required");
		return value == MatrixSide.LEFT ? LEFT : RIGHT;
	}
	private static void checkRegion(int count, double[] vector, int offset, int stride) {
		if (count < 0 || vector == null || offset < 0 || stride < 1
				|| (count > 0 && offset + (long) (count - 1) * stride >= vector.length))
			throw new IllegalArgumentException("invalid vector region");
	}
	private static void checkRegion(int count, float[] vector, int offset, int stride) {
		if (count < 0 || vector == null || offset < 0 || stride < 1
				|| (count > 0 && offset + (long) (count - 1) * stride >= vector.length))
			throw new IllegalArgumentException("invalid vector region");
	}
	private static void checkGemv(MatrixTranspose transpose, int rows, int columns,
			double[] matrix, double[] x, double[] y) {
		trans(transpose); checkMatrix(matrix, rows, columns);
		int input = transpose == MatrixTranspose.NONE ? columns : rows;
		int output = transpose == MatrixTranspose.NONE ? rows : columns;
		if (x == null || x.length != input || y == null || y.length != output)
			throw new IllegalArgumentException("invalid GEMV dimensions");
	}
	private static void checkGemv(MatrixTranspose transpose, int rows, int columns,
			float[] matrix, float[] x, float[] y) {
		trans(transpose); checkMatrix(matrix, rows, columns);
		int input = transpose == MatrixTranspose.NONE ? columns : rows;
		int output = transpose == MatrixTranspose.NONE ? rows : columns;
		if (x == null || x.length != input || y == null || y.length != output)
			throw new IllegalArgumentException("invalid GEMV dimensions");
	}
	private static void checkGemm(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared,
			double[] left, double[] right, double[] result) {
		trans(leftTranspose); trans(rightTranspose);
		int leftRows = leftTranspose == MatrixTranspose.NONE ? rows : shared;
		int leftColumns = leftTranspose == MatrixTranspose.NONE ? shared : rows;
		int rightRows = rightTranspose == MatrixTranspose.NONE ? shared : columns;
		int rightColumns = rightTranspose == MatrixTranspose.NONE ? columns : shared;
		checkMatrix(left, leftRows, leftColumns); checkMatrix(right, rightRows, rightColumns);
		checkMatrix(result, rows, columns);
	}
	private static void checkGemm(MatrixTranspose leftTranspose,
			MatrixTranspose rightTranspose, int rows, int columns, int shared,
			float[] left, float[] right, float[] result) {
		trans(leftTranspose); trans(rightTranspose);
		int leftRows = leftTranspose == MatrixTranspose.NONE ? rows : shared;
		int leftColumns = leftTranspose == MatrixTranspose.NONE ? shared : rows;
		int rightRows = rightTranspose == MatrixTranspose.NONE ? shared : columns;
		int rightColumns = rightTranspose == MatrixTranspose.NONE ? columns : shared;
		checkMatrix(left, leftRows, leftColumns); checkMatrix(right, rightRows, rightColumns);
		checkMatrix(result, rows, columns);
	}
	private static void checkSyrk(MatrixTranspose transpose, int dimension, int shared,
			double[] matrix, double[] result) {
		trans(transpose); checkMatrix(matrix, transpose == MatrixTranspose.NONE ? dimension : shared,
				transpose == MatrixTranspose.NONE ? shared : dimension);
		checkSquare(result, dimension);
	}
	private static void checkSyrk(MatrixTranspose transpose, int dimension, int shared,
			float[] matrix, float[] result) {
		trans(transpose); checkMatrix(matrix, transpose == MatrixTranspose.NONE ? dimension : shared,
				transpose == MatrixTranspose.NONE ? shared : dimension);
		checkSquare(result, dimension);
	}
	private static void checkTriangular(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector,
			int vectorLength) {
		triangle(triangle); trans(transpose); diagonal(diagonal); checkSquare(matrix, dimension);
		if (vector == null || vector.length != vectorLength)
			throw new IllegalArgumentException("invalid triangular solve dimensions");
	}
	private static void checkTriangular(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector,
			int vectorLength) {
		triangle(triangle); trans(transpose); diagonal(diagonal); checkSquare(matrix, dimension);
		if (vector == null || vector.length != vectorLength)
			throw new IllegalArgumentException("invalid triangular solve dimensions");
	}
	private static void checkTrsm(MatrixSide sideValue, MatrixTriangle triangleValue,
			MatrixTranspose transpose, MatrixDiagonal diagonalValue, int rows, int columns,
			double[] matrix, double[] right) {
		int order = sideValue == MatrixSide.LEFT ? rows : columns;
		side(sideValue); triangle(triangleValue); trans(transpose); diagonal(diagonalValue);
		checkSquare(matrix, order); checkMatrix(right, rows, columns);
	}
	private static void checkTrsm(MatrixSide sideValue, MatrixTriangle triangleValue,
			MatrixTranspose transpose, MatrixDiagonal diagonalValue, int rows, int columns,
			float[] matrix, float[] right) {
		int order = sideValue == MatrixSide.LEFT ? rows : columns;
		side(sideValue); triangle(triangleValue); trans(transpose); diagonal(diagonalValue);
		checkSquare(matrix, order); checkMatrix(right, rows, columns);
	}
	private static void checkMatrix(double[] matrix, int rows, int columns) {
		if (rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns)
			throw new IllegalArgumentException("invalid matrix dimensions");
	}
	private static void checkMatrix(float[] matrix, int rows, int columns) {
		if (rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns)
			throw new IllegalArgumentException("invalid matrix dimensions");
	}
	private static void checkSquare(double[] matrix, int dimension) {
		checkMatrix(matrix, dimension, dimension);
	}
	private static void checkSquare(float[] matrix, int dimension) {
		checkMatrix(matrix, dimension, dimension);
	}
	private static void mirrorLower(double[] matrix, int dimension) {
		for (int row = 0; row < dimension; row++) for (int column = row + 1;
				column < dimension; column++) matrix[row * dimension + column]
					= matrix[column * dimension + row];
	}
	private static void mirrorLower(float[] matrix, int dimension) {
		for (int row = 0; row < dimension; row++) for (int column = row + 1;
				column < dimension; column++) matrix[row * dimension + column]
					= matrix[column * dimension + row];
	}
	private static void clearUpper(double[] matrix, int dimension) {
		for (int row = 0; row < dimension; row++) for (int column = row + 1;
				column < dimension; column++) matrix[row * dimension + column] = 0.0;
	}
	private static void clearUpper(float[] matrix, int dimension) {
		for (int row = 0; row < dimension; row++) for (int column = row + 1;
				column < dimension; column++) matrix[row * dimension + column] = 0.0f;
	}
	private static void zeroBase(int[] pivot) {
		for (int i = 0; i < pivot.length; i++) pivot[i]--;
	}
	private static void checkInfo(String operation, int info) {
		if (info < 0) throw new IllegalArgumentException(operation + " rejected argument " + -info);
		if (info > 0) throw new ArithmeticException(operation + " failed to converge/factor at " + info);
	}
	static List<File> oneApiLibraries() {
		String root = System.getenv("ONEAPI_ROOT");
		if (root == null || root.isEmpty()) return Collections.emptyList();
		File mkl = new File(root, "mkl"); File[] versions = mkl.listFiles();
		if (versions == null) return Collections.emptyList();
		List<File> ordered = new ArrayList<File>(); Collections.addAll(ordered, versions);
		Collections.sort(ordered, new Comparator<File>() {
			@Override public int compare(File first, File second) {
				return second.getName().compareTo(first.getName());
			}
		});
		List<File> result = new ArrayList<File>();
		for (File version : ordered) {
			File windows = new File(version, "redist/intel64/mkl_rt.2.dll");
			File linux = new File(version, "lib/intel64/libmkl_rt.so");
			File mac = new File(version, "lib/libmkl_rt.dylib");
			if (windows.isFile()) result.add(windows);
			if (linux.isFile()) result.add(linux);
			if (mac.isFile()) result.add(mac);
		}
		return result;
	}
}
