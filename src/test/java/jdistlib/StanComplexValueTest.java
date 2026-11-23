/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.math.Complex;
import jdistlib.rng.MersenneTwister;

public class StanComplexValueTest {
	@Test public void javaComplexImplementsPrincipalOperations() {
		Complex value = new Complex(3,4);
		assertEquals(5,value.abs(),0);
		assertEquals(value.real(),value.log().exp().real(),1e-12);
		assertEquals(value.imaginary(),value.sqrt().multiply(value.sqrt()).imaginary(),1e-12);
	}

	@Test public void complexParameterArithmeticRunsOnReverseTape() {
		String source = "parameters { complex z; } model { complex target_value = 1+2i; "
				+ "target += -norm(z-target_value); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertEquals(2,model.dimension());
		assertTrue(Gradients.check(model,new double[]{.2,-.4},2e-6,2e-6).passed());
		double[] gradient=new double[2];model.logDensityAndGradient(new double[]{.2,-.4},gradient);
		assertEquals(1.6,gradient[0],1e-12);assertEquals(4.8,gradient[1],1e-12);
	}

	@Test public void complexTangentRetainsLargeImaginaryLimits() {
		// R r90957 / PR#19327: finite arguments plus analytic infinite limits.
		for(double y:new double[]{30,350,400,Double.POSITIVE_INFINITY}) for(int sign:new int[]{-1,1}) {
			Complex value=new Complex(.7,sign*y).tan();
			double expected=2*Math.sin(1.4)*Math.exp(-2*y);
			assertEquals(expected,value.real(),Math.abs(expected)*2e-14);
			assertEquals(sign,value.imaginary(),0);
			String source="data { complex z; } parameters { real x; } model { x ~ normal(0,1); } "
					+"generated quantities { complex result = tan(z); }";
			CompiledModelScript compiled=ModelScript.compileStan(source,Collections.singletonMap("z",new double[]{.7,sign*y}));
			double[] result=compiled.generate(new double[]{0},new MersenneTwister(1)).get("result");
			assertEquals(value.real(),result[0],0);assertEquals(value.imaginary(),result[1],0);
		}
		BayesianModel model=ModelScript.compileStan("parameters { complex z; } model { target += get_real(tan(z)); }").model();
		double[] gradient=new double[2];
		model.logDensityAndGradient(new double[]{.7,30},gradient);
		assertEquals(4*Math.cos(1.4)*Math.exp(-60),gradient[0],1e-39);
		assertEquals(-4*Math.sin(1.4)*Math.exp(-60),gradient[1],1e-39);
	}

	@Test public void complexVectorsUseConjugateTransposeAndGeneratedInterleaving() {
		String source = "parameters { complex_vector[2] z; } transformed parameters { complex energy = z' * z; } "
				+ "model { target += -get_real(energy); } generated quantities { complex copy = conj(z[1]); }";
		CompiledModelScript compiled=ModelScript.compileStan(source);
		BayesianModel model=compiled.model();assertEquals(4,model.dimension());
		assertTrue(Gradients.check(model,new double[]{1,2,-.5,.25},2e-6,2e-6).passed());
		double[] copy=compiled.generate(new double[]{1,2,-.5,.25},new MersenneTwister(1)).get("copy");
		assertEquals(1,copy[0],0);assertEquals(-2,copy[1],0);
	}

	@Test public void complexDataUsesInterleavedRealImaginaryStorage() {
		String source="data { array[2] complex z; } parameters { real x; } model { target += x*sum(get_real(z)); x~normal(0,1); }";
		BayesianModel model=ModelScript.compileStan(source,Collections.singletonMap("z",new double[]{1,2,3,4})).model();
		double[] gradient=new double[1];model.logDensityAndGradient(new double[]{0},gradient);assertEquals(4,gradient[0],1e-12);
	}
}
