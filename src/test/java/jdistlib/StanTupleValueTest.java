/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ModelScript;
import jdistlib.inference.lang.TupleValue;

public class StanTupleValueTest {
	@Test public void tupleMembersAndNestedAssignmentRemainDifferentiable() {
		String source = "parameters { real x; } model { "
				+ "tuple(real, tuple(vector[2], complex)) t = (x, ([x, 2*x]', 1+2i)); "
				+ "t.2.1 = [2*x, 3*x]'; target += t.1 + sum(t.2.1) + get_real(t.2.2); "
				+ "x ~ normal(0, 1); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		double[] gradient = new double[1];
		model.logDensityAndGradient(new double[] {.2}, gradient);
		assertEquals(5.8, gradient[0], 1e-12);
		assertTrue(Gradients.check(model, new double[] {.2}, 2e-6, 2e-6).passed());
	}

	@Test public void tupleArgumentsAndReturnsParticipateInOverloadResolution() {
		String source = "functions { tuple(real, vector) scale(tuple(real, vector) x, real a) "
				+ "{ return (x.1*a, x.2*a); } real total(tuple(real, vector) x) "
				+ "{ return x.1 + sum(x.2); } } parameters { real x; } model { "
				+ "tuple(real, vector[2]) input = (x, [x, 2*x]'); "
				+ "target += total(scale(input, 2)); x ~ normal(0, 1); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, new double[] {.4}, 2e-6, 2e-6).passed());
		double[] gradient = new double[1]; model.logDensityAndGradient(new double[] {.4}, gradient);
		assertEquals(7.6, gradient[0], 1e-12);
	}

	@Test public void publicTupleValueIsImmutableAndOneBased() {
		TupleValue original = new TupleValue(1.0, new double[] {2, 3});
		TupleValue changed = original.withMember(1, 4.0);
		assertEquals(1.0, (Double) original.member(1), 0);
		assertEquals(4.0, (Double) changed.member(1), 0);
		assertTrue(Arrays.equals(new double[] {2, 3}, (double[]) changed.member(2)));
	}
}
