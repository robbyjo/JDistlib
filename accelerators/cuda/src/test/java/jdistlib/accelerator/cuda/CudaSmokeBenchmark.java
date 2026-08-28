/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.cuda;

import java.util.Locale;

import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.CpuComputeBackend;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.PreparedLogisticRegression;

/** Reproducible CPU/CUDA likelihood smoke benchmark; not a general-purpose microbenchmark. */
public final class CudaSmokeBenchmark {
	private CudaSmokeBenchmark() {}
	public static void main(String[] arguments) {
		Locale.setDefault(Locale.ROOT);
		CudaComputeBackend cuda = new CudaComputeBackend();
		if (!cuda.available()) throw new IllegalStateException("CUDA unavailable", cuda.unavailableCause());
		CpuComputeBackend cpu = new CpuComputeBackend();
		int rows = 8192, dimensions = 32;
		double[][] design = new double[rows][dimensions]; double[] outcomes = new double[rows];
		for (int i = 0; i < rows; i++) {
			outcomes[i] = ((i * 17 + 3) & 7) < 4 ? 1.0 : 0.0;
			for (int j = 0; j < dimensions; j++) design[i][j] = Math.sin((i + 1.0) * (j + 3.0) * 0.001);
		}
		System.out.println("backend=" + cuda.capabilities().device());
		System.out.println("rows\tdimensions\tchains\tcpu_ms\tcuda_resident_ms\tcuda_end_to_end_ms\tspeedup_resident\tspeedup_end_to_end\tmax_abs_error");
		try {
			for (int chains : new int[] {1, 4, 16, 64}) benchmark(cpu, cuda, design, outcomes, chains);
		} finally { cuda.close(); }
	}
	private static void benchmark(ComputeBackend cpu, ComputeBackend cuda, double[][] design,
			double[] outcomes, int chains) {
		double[][] states = new double[chains][design[0].length];
		for (int c = 0; c < chains; c++) for (int j = 0; j < states[c].length; j++)
			states[c][j] = 0.02 * Math.cos((c + 2.0) * (j + 1.0));
		PreparedLogisticRegression cpuPrepared = cpu.prepareLogisticRegression(design, outcomes);
		PreparedLogisticRegression cudaPrepared = cuda.prepareLogisticRegression(design, outcomes);
		int warmup = 5, repetitions = chains <= 4 ? 20 : 8;
		try {
			for (int i = 0; i < warmup; i++) { cpuPrepared.evaluate(states, 0.1); cudaPrepared.evaluate(states, 0.1); }
			long cpuNanos = medianPrepared(cpuPrepared, states, repetitions);
			long cudaNanos = medianPrepared(cudaPrepared, states, repetitions);
			long endToEndNanos = medianEndToEnd(cuda, design, outcomes, states, repetitions);
			LogisticRegressionBatchResult expected = cpuPrepared.evaluate(states, 0.1);
			LogisticRegressionBatchResult actual = cudaPrepared.evaluate(states, 0.1);
			double error = maximumError(expected, actual);
			double cpuMs = cpuNanos / 1e6 / repetitions, cudaMs = cudaNanos / 1e6 / repetitions;
			double endMs = endToEndNanos / 1e6 / repetitions;
			System.out.printf("%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.3f\t%.3f\t%.3g%n",
					design.length, design[0].length, chains, cpuMs, cudaMs, endMs,
					cpuMs / cudaMs, cpuMs / endMs, error);
		} finally { cpuPrepared.close(); cudaPrepared.close(); }
	}
	private static long timePrepared(PreparedLogisticRegression prepared, double[][] states, int repetitions) {
		long start = System.nanoTime(); double checksum = 0.0;
		for (int i = 0; i < repetitions; i++) checksum += prepared.evaluate(states, 0.1).logDensities()[0];
		long elapsed = System.nanoTime() - start; if (!Double.isFinite(checksum)) throw new AssertionError(); return elapsed;
	}
	private static long medianPrepared(PreparedLogisticRegression prepared, double[][] states, int repetitions) {
		long[] trials = new long[5]; for (int i = 0; i < trials.length; i++) trials[i] = timePrepared(prepared, states, repetitions);
		java.util.Arrays.sort(trials); return trials[trials.length / 2];
	}
	private static long timeEndToEnd(ComputeBackend backend, double[][] design, double[] outcomes,
			double[][] states, int repetitions) {
		long start = System.nanoTime(); double checksum = 0.0;
		for (int i = 0; i < repetitions; i++) checksum += backend.logisticRegression(design, outcomes, states, 0.1).logDensities()[0];
		long elapsed = System.nanoTime() - start; if (!Double.isFinite(checksum)) throw new AssertionError(); return elapsed;
	}
	private static long medianEndToEnd(ComputeBackend backend, double[][] design, double[] outcomes,
			double[][] states, int repetitions) {
		long[] trials = new long[5]; for (int i = 0; i < trials.length; i++) trials[i] = timeEndToEnd(backend, design, outcomes, states, repetitions);
		java.util.Arrays.sort(trials); return trials[trials.length / 2];
	}
	private static double maximumError(LogisticRegressionBatchResult first, LogisticRegressionBatchResult second) {
		double result = 0.0; double[] a = first.logDensities(), b = second.logDensities();
		for (int i = 0; i < a.length; i++) result = Math.max(result, Math.abs(a[i] - b[i]));
		double[][] ga = first.gradients(), gb = second.gradients();
		for (int i = 0; i < ga.length; i++) for (int j = 0; j < ga[i].length; j++)
			result = Math.max(result, Math.abs(ga[i][j] - gb[i][j]));
		return result;
	}
}
