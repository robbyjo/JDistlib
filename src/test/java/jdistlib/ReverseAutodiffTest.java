/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.autodiff.ReverseModeGradient;
import jdistlib.inference.autodiff.ReverseModeLogDensity;
import jdistlib.inference.autodiff.ReverseTape;

public class ReverseAutodiffTest {
	@Test public void atomicKernelUsesReusableEdgeArena() {
		ReverseTape tape = new ReverseTape(4);
		int x = tape.variable(2), y = tape.variable(3);
		int productSum = tape.atomic(13, new int[] {x, y}, new double[] {4, 3});
		tape.reverse(productSum);
		assertEquals(4, tape.adjoint(x), 0);
		assertEquals(3, tape.adjoint(y), 0);
		int mark = tape.mark();
		tape.atomic(1, new int[] {x}, new double[] {7});
		tape.rewind(mark);
		int replacement = tape.atomic(5, new int[] {y}, new double[] {2});
		tape.reverse(replacement);
		assertEquals(2, tape.adjoint(y), 0);
	}
	@Test public void differentiatesCoupledExpression() {
		ReverseModeGradient engine = new ReverseModeGradient(new ReverseTape(2));
		double[] gradient = new double[2];
		double value = engine.evaluate((tape, x) -> {
			int product = tape.multiply(x[0], x[1]);
			return tape.add(tape.exp(product), tape.sin(x[0]));
		}, new double[] {0.7, -0.2}, gradient);
		double exponential = Math.exp(-0.14);
		assertEquals(exponential + Math.sin(0.7), value, 1e-12);
		assertEquals(-0.2 * exponential + Math.cos(0.7), gradient[0], 1e-12);
		assertEquals(0.7 * exponential, gradient[1], 1e-12);
		assertTrue(engine.tape().capacity() > 2);
	}

	@Test public void resetAndRewindReuseArenaWithoutStaleAdjoints() {
		ReverseTape tape = new ReverseTape(4);
		int x = tape.variable(3.0); int mark = tape.mark();
		int square = tape.multiply(x, x); tape.reverse(square);
		assertEquals(6.0, tape.adjoint(x), 0.0);
		tape.rewind(mark);
		int linear = tape.multiply(x, 5.0); tape.reverse(linear);
		assertEquals(5.0, tape.adjoint(x), 0.0);
		int capacity = tape.capacity(); tape.reset();
		assertEquals(0, tape.size()); assertEquals(capacity, tape.capacity());
	}

	@Test public void reusableLogDensityMatchesForwardModeModel() {
		jdistlib.inference.BayesianModel forward = jdistlib.inference.lang.ModelScript.compile(
				"parameters { vector[3] x; } model { x ~ normal(0,1); target += square(sum(x))/6; }").model();
		ReverseModeLogDensity reverse = new ReverseModeLogDensity(3, (tape, x) -> {
			int result = tape.constant(-1.5 * Math.log(2.0 * Math.PI));
			int sum = tape.constant(0.0);
			for (int value : x) {
				result = tape.subtract(result, tape.multiply(tape.multiply(value, value), 0.5));
				sum = tape.add(sum, value);
			}
			return tape.add(result, tape.divide(tape.multiply(sum, sum), 6.0));
		});
		double[] position = {0.2, -0.4, 0.7};
		double[] forwardGradient = new double[3], reverseGradient = new double[3];
		double forwardValue = forward.logDensityAndGradient(position, forwardGradient);
		double reverseValue = reverse.logDensityAndGradient(position, reverseGradient);
		assertEquals(forwardValue, reverseValue, 1e-12);
		for (int i = 0; i < 3; i++) assertEquals(forwardGradient[i], reverseGradient[i], 1e-12);
		int capacity = reverse.tape().capacity();
		reverse.logDensityAndGradient(position, reverseGradient);
		assertEquals(capacity, reverse.tape().capacity());
	}
}
