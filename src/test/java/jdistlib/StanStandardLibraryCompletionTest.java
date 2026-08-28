/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ModelScript;

public class StanStandardLibraryCompletionTest {
	@Test public void reductionAndDistanceOverloadsDifferentiate() {
		String source = "parameters { vector[3] x; } model { vector[3] y = [1,2,4]'; "
				+ "target += dot_self(x) + distance(x,y) + squared_distance(x,y) + mean(x) "
				+ "+ variance(x) + sd(x) + log_sum_exp(x) + min(x) + max(x); "
				+ "x ~ normal(0,2); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, new double[] {.2, 1.1, 2.3}, 3e-6, 3e-6).passed());
	}

	@Test public void spdAndQuadraticMatrixOverloadsDifferentiate() {
		String source = "parameters { real x; } model { "
				+ "matrix[2,2] L = [[exp(x),0],[x,2]]; "
				+ "matrix[2,2] K = multiply_lower_tri_self_transpose(L); "
				+ "matrix[2,2] Ki = inverse_spd(K); matrix[2,2] I = mdivide_left_spd(K,K); "
				+ "matrix[2,2] R = mdivide_right_spd(K,K); "
				+ "target += log_determinant_spd(K) + trace(Ki) + trace_quad_form(K,I) "
				+ "+ trace_gen_quad_form(I,K,I) + sum(columns_dot_self(K)) "
				+ "+ sum(rows_dot_self(K)) + sum(symmetrize_from_lower_tri(L)) + sum(R); "
				+ "x ~ normal(0,1); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, new double[] {.15}, 8e-6, 8e-6).passed());
	}

	@Test public void complexTrigonometricAndHyperbolicOverloadsDifferentiate() {
		String source = "parameters { complex z; } model { complex a = sin(z)+cos(z)+tan(z); "
				+ "complex b = sinh(z)+cosh(z)+tanh(z); target += get_real(a+b)-norm(z); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, new double[] {.2, -.3}, 5e-6, 5e-6).passed());
	}
}
