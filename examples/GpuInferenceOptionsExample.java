/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.inference.AcceleratedLogisticRegression;
import jdistlib.inference.InferenceCliOptions;
import jdistlib.inference.SamplingOptions;

/** Programmatic and command-line accelerator selection for an inference target. */
public final class GpuInferenceOptionsExample {
	private GpuInferenceOptionsExample() {}
	public static void main(String[] arguments) {
		InferenceCliOptions cli = InferenceCliOptions.parse(arguments);
		SamplingOptions options = cli.applyTo(SamplingOptions.builder()
				.warmupIterations(500).sampleIterations(1000)).build();
		double[][] design = {{1.0, -1.0}, {1.0, 0.0}, {1.0, 1.0}};
		double[] outcomes = {0.0, 0.0, 1.0};
		try (AcceleratedLogisticRegression target = AcceleratedLogisticRegression.forNuts(
				options, design, outcomes, 1.0)) {
			System.out.println("compute backend=" + target.computeBackend().id()
					+ ", device=" + target.computeBackend().capabilities().device()
					+ ", NUTS offload=" + options.nutsBackend());
		}
	}
}
