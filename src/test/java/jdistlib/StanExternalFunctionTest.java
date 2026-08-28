/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ExternalFunctionRegistry;
import jdistlib.inference.lang.ExternalFunctionResult;
import jdistlib.inference.lang.ModelScript;

public class StanExternalFunctionTest {
	@Test public void externalFunctionProvidesExactReverseJacobian() {
		ExternalFunctionRegistry registry = ExternalFunctionRegistry.builder()
				.bind("java_penalty", arguments -> {
					double x = arguments[0][0], scale = arguments[1][0];
					return ExternalFunctionResult.scalar(scale*x*x, 2*scale*x, x*x);
				}).build();
		String source = "functions { real java_penalty(real x, real scale); } "
				+ "parameters { real x; real scale; } model { target += -java_penalty(x,scale); "
				+ "x ~ normal(0,1); scale ~ normal(1,1); }";
		BayesianModel model = ModelScript.compileStan(source,
				Collections.<String,double[]>emptyMap(), registry).model();
		assertTrue(Gradients.check(model, new double[] {.3,1.2}, 2e-6, 2e-6).passed());
		double[] gradient = new double[2]; model.logDensityAndGradient(new double[] {.3,1.2}, gradient);
		assertEquals(-.3-2*1.2*.3, gradient[0], 1e-12);
	}

	@Test public void externalVectorResultChainsThroughDotProduct() {
		ExternalFunctionRegistry registry = ExternalFunctionRegistry.builder()
				.bind("java_scale", arguments -> new ExternalFunctionResult(
						new double[] {arguments[0][0]*arguments[1][0], arguments[0][1]*arguments[1][0]},
						new int[] {2}, new double[][] {
							{arguments[1][0],0,arguments[0][0]}, {0,arguments[1][0],arguments[0][1]}
						})).build();
		String source = "functions { vector java_scale(vector x, real scale); } "
				+ "parameters { vector[2] x; real scale; } model { vector[2] y = java_scale(x,scale); "
				+ "target += dot_product(y,y); x ~ normal(0,1); scale ~ normal(0,1); }";
		assertTrue(Gradients.check(ModelScript.compileStan(source,
				Collections.<String,double[]>emptyMap(), registry).model(),
				new double[] {.2,-.3,.7}, 2e-6, 2e-6).passed());
	}

	@Test public void externalFunctionFlattensNestedTupleArguments() {
		ExternalFunctionRegistry registry = ExternalFunctionRegistry.builder()
				.bind("java_tuple_sum", arguments -> ExternalFunctionResult.scalar(
						arguments[0][0]+arguments[0][1]+arguments[0][2], 1,1,1))
				.build();
		String source = "functions { real java_tuple_sum(tuple(real, vector) x); } "
				+ "parameters { real x; } model { tuple(real, vector[2]) t=(x,[2*x,3*x]'); "
				+ "target += java_tuple_sum(t)-square(x); }";
		BayesianModel model = ModelScript.compileStan(source,
				Collections.<String,double[]>emptyMap(), registry).model();
		assertTrue(Gradients.check(model, new double[] {.3}, 2e-6, 2e-6).passed());
	}
}
