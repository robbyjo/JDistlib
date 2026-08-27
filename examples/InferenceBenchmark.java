/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Constraints;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelEvaluator;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.SamplingOptions;
import jdistlib.rng.MersenneTwister;

/** Reproducible smoke benchmark; use JMH for publication-quality measurements. */
public final class InferenceBenchmark {
	private InferenceBenchmark() {}
	public static void main(String[] arguments) {
		BayesianModel model = new ModelBuilder().data("y", observations())
				.parameter("mu", Constraints.real(), 0.0)
				.factor("prior", new String[] {"mu"}, ModelFactors.normalPrior("mu", 0, 5))
				.factor("likelihood", new String[] {"mu", "y"},
						ModelFactors.normalObservations("y", "mu", 1)).build();
		ModelEvaluator evaluator = model.evaluator();
		double[] state = model.initialState();
		double[] gradient = new double[state.length];
		for (int i = 0; i < 10_000; i++) evaluator.logDensityAndGradient(state, gradient);
		long start = System.nanoTime();
		for (int i = 0; i < 100_000; i++) evaluator.logDensityAndGradient(state, gradient);
		long evaluationNanos = System.nanoTime() - start;
		start = System.nanoTime();
		new NoUTurnSampler().sample(model, state, SamplingOptions.builder()
				.warmupIterations(250).sampleIterations(500).build(), new MersenneTwister(42));
		long samplingNanos = System.nanoTime() - start;
		System.out.println("100k vectorized gradient evaluations ns=" + evaluationNanos
				+ " evaluations/s=" + (100_000_000_000_000L / evaluationNanos));
		System.out.println("NUTS warmup+sampling ns=" + samplingNanos
				+ " transitions/s=" + (750_000_000_000L / samplingNanos));
	}
	private static double[] observations() {
		double[] values = new double[1000];
		for (int i = 0; i < values.length; i++) values[i] = Math.sin(i * 0.1);
		return values;
	}
}
