/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.autodiff.ReverseModeLogDensity;
import jdistlib.inference.lang.ModelScript;

/** Small repeatable forward-versus-reverse gradient throughput benchmark. */
public final class ReverseAutodiffBenchmark {
	private ReverseAutodiffBenchmark() {}

	public static void main(String[] arguments) {
		BayesianModel forward = ModelScript.compile("parameters { vector[8] x; } model { "
				+ "x ~ normal(0,1); target += square(sum(x))/16; }").model();
		ReverseModeLogDensity reverse = new ReverseModeLogDensity(8, (tape, x) -> {
			int result = tape.constant(-8 * 0.5 * Math.log(2 * Math.PI));
			int sum = tape.constant(0.0);
			for (int value : x) {
				result = tape.subtract(result, tape.multiply(tape.multiply(value, value), 0.5));
				sum = tape.add(sum, value);
			}
			return tape.add(result, tape.divide(tape.multiply(sum, sum), 16.0));
		});
		double[] position = {.1, -.2, .3, -.4, .5, -.6, .7, -.8};
		double[] gradient = new double[position.length]; int repetitions = 100000;
		for (int i = 0; i < 2000; i++) {
			forward.logDensityAndGradient(position, gradient);
			reverse.logDensityAndGradient(position, gradient);
		}
		long start = System.nanoTime();
		for (int i = 0; i < repetitions; i++) forward.logDensityAndGradient(position, gradient);
		long forwardNanos = System.nanoTime() - start;
		start = System.nanoTime();
		for (int i = 0; i < repetitions; i++) reverse.logDensityAndGradient(position, gradient);
		long reverseNanos = System.nanoTime() - start;
		System.out.printf("forward %.1f ns/eval, reverse %.1f ns/eval, reverse tape capacity %d%n",
				forwardNanos / (double) repetitions, reverseNanos / (double) repetitions,
				reverse.tape().capacity());
	}
}
