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
