/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import java.util.Locale;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedCsrMatrix;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.matrix.CsrMatrix;

/** Reproducible prepared-CSR and sparse-refactorization comparison benchmark. */
public final class NativeSparseBenchmark {
	private NativeSparseBenchmark() {}
	public static void main(String[] arguments) {
		Locale.setDefault(Locale.ROOT); int dimension = integerProperty("dimension", 4000);
		int repetitions = integerProperty("repetitions", 20);
		CsrMatrix matrix = tridiagonal(dimension, 4.0); double[] vector = new double[dimension];
		for (int i = 0; i < dimension; i++) vector[i] = Math.sin(i * 0.01);
		System.out.println("dimension\tnnz\tbackend\tprepared_csr_mv_ms\tinitial_factor_ms"
				+ "\trefactor_solve_ms\tchecksum");
		benchmark(new CpuComputeBackend(), matrix, vector, repetitions);
		try (OneMklComputeBackend oneMkl = new OneMklComputeBackend()) {
			if (oneMkl.available() && oneMkl.capabilities().nativeSparseFactorizations())
				benchmark(oneMkl, matrix, vector, repetitions);
			else System.out.println("# oneMKL PARDISO unavailable: " + oneMkl.unavailableCause());
		}
		try (OpenBlasComputeBackend openBlas = new OpenBlasComputeBackend()) {
			if (openBlas.available() && openBlas.capabilities().nativeSparseFactorizations())
				benchmark(openBlas, matrix, vector, repetitions);
			else System.out.println("# OpenBLAS/CHOLMOD unavailable: "
					+ (openBlas.available() ? openBlas.cholmodUnavailableCause()
							: openBlas.unavailableCause()));
		}
	}
	private static void benchmark(ComputeBackend backend, CsrMatrix matrix, double[] vector,
			int repetitions) {
		double[] output = new double[matrix.rows()]; long productStart;
		try (PreparedCsrMatrix prepared = backend.prepareDcsr(matrix)) {
			for (int i = 0; i < 5; i++) prepared.multiply(1.0, vector, 0.0, output);
			productStart = System.nanoTime();
			for (int i = 0; i < repetitions; i++) prepared.multiply(1.0, vector, 0.0, output);
		}
		long productNanos = System.nanoTime() - productStart;
		long factorStart = System.nanoTime();
		PreparedSparseCholesky factor = backend.prepareDcsrpotrf(matrix, MatrixTriangle.LOWER);
		long initialNanos = System.nanoTime() - factorStart; double checksum = 0.0;
		long refactorStart = System.nanoTime();
		try {
			for (int iteration = 0; iteration < repetitions; iteration++) {
				factor.refactor(tridiagonal(matrix.rows(), 4.0 + iteration * 1e-4));
				double[] solution = factor.solve(vector); checksum += solution[iteration % solution.length];
			}
		} finally { factor.close(); }
		long refactorNanos = System.nanoTime() - refactorStart;
		for (double value : output) checksum += value;
		System.out.printf("%d\t%d\t%s\t%.4f\t%.4f\t%.4f\t%.12g%n", matrix.rows(),
				matrix.nonzeroCount(), backend.id(), productNanos / 1e6 / repetitions,
				initialNanos / 1e6, refactorNanos / 1e6 / repetitions, checksum);
	}
	private static CsrMatrix tridiagonal(int dimension, double diagonal) {
		double[] values = new double[2 * dimension - 1]; int[] columns = new int[values.length];
		int[] starts = new int[dimension + 1]; int offset = 0; starts[0] = 1;
		for (int row = 0; row < dimension; row++) {
			if (row > 0) { values[offset] = -1.0; columns[offset++] = row; }
			values[offset] = diagonal; columns[offset++] = row + 1; starts[row + 1] = offset + 1;
		}
		return new CsrMatrix(dimension, dimension, values, columns, starts);
	}
	private static int integerProperty(String name, int fallback) {
		String value = System.getProperty("jdistlib.benchmark." + name);
		if (value == null) return fallback; int parsed = Integer.parseInt(value);
		if (parsed < 1) throw new IllegalArgumentException(name + " must be positive"); return parsed;
	}
}
