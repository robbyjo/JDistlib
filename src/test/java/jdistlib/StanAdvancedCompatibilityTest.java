/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ModelScript;
import jdistlib.inference.lang.ModelScriptException;

public class StanAdvancedCompatibilityTest {
	@Test public void multidimensionalSlicesAndIndexedAssignmentsDifferentiate() {
		String source = "data { array[2,3] real x; } parameters { real theta; } model { "
				+ "array[2,3] real work = x; work[1,2:3] = rep_array(theta,2); "
				+ "work[:,1] += theta; target += sum(work); theta ~ normal(0,1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("x", new double[] {1, 2, 3, 4, 5, 6});
		BayesianModel model = ModelScript.compileStan(source, data).model();
		double[] gradient = new double[1];
		model.logDensityAndGradient(new double[] {0.2}, gradient);
		assertEquals(3.8, gradient[0], 1e-12);
		assertTrue(Gradients.check(model, new double[] {0.2}, 2e-6, 2e-6).passed());
	}

	@Test public void arraysOfVectorsAndMatricesRetainBaseTypes() {
		String source = "data { array[2] vector[3] v; array[2] matrix[2,2] A; } "
				+ "parameters { vector[3] beta; } model { beta ~ normal(0,1); "
				+ "target += dot_product(v[2], beta) + sum(A[1,1:2,2]); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("v", new double[] {1, 2, 3, 4, 5, 6});
		data.put("A", new double[] {1, 2, 3, 4, 5, 6, 7, 8});
		BayesianModel model = ModelScript.compileStan(source, data).model();
		double[] gradient = new double[3];
		model.logDensityAndGradient(new double[] {0, 0, 0}, gradient);
		assertEquals(4.0, gradient[0], 1e-12);
		assertEquals(5.0, gradient[1], 1e-12);
		assertEquals(6.0, gradient[2], 1e-12);
	}

	@Test public void matrixAlgebraDecompositionsAndProbabilityKernelsDifferentiate() {
		String source = "data { matrix[3,2] X; vector[3] y; matrix[3,3] Sigma; } "
				+ "parameters { vector[2] beta; } transformed parameters { vector[3] mu = X * beta; } "
				+ "model { beta ~ normal(0,2); y ~ multi_normal(mu, Sigma); "
				+ "target += 0 * log_determinant(Sigma) + 0 * determinant(inverse(Sigma)); "
				+ "target += 0 * sum(mdivide_left_spd(Sigma,y)); "
				+ "target += 0 * (sum(qr_thin_Q(X)) + sum(qr_thin_R(X))); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("X", new double[] {1, 0, 1, 1, 0, 1});
		data.put("y", new double[] {0.2, -0.1, 0.5});
		data.put("Sigma", new double[] {1.2, .2, .1, .2, 1.1, .15, .1, .15, .9});
		BayesianModel model = ModelScript.compileStan(source, data).model();
		GradientCheckResult check = Gradients.check(model, new double[] {0.1, -0.2}, 2e-5, 2e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void typedFunctionsProbabilitySuffixesAndLpEffectsExecute() {
		String source = "functions { "
				+ "vector center(vector x, real mu) { return x - mu; } "
				+ "real score(data matrix A, vector b) { return sum(A*b); } "
				+ "real custom_lpdf(vector y, real mu) { return normal_lpdf(y | mu, 1); } "
				+ "real regularize_lp(real x) { target += normal_lpdf(x | 0, 2); return 0; } "
				+ "} data { vector[3] y; matrix[3,2] X; } parameters { vector[2] beta; real mu; } "
				+ "model { center(y,mu) ~ custom(0); target += 0.1*score(X,beta); "
				+ "target += regularize_lp(mu); beta ~ normal(0,1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("y", new double[] {0.2, -0.1, 0.4});
		data.put("X", new double[] {1, 0, 1, 1, 0, 1});
		BayesianModel model = ModelScript.compileStan(source, data).model();
		GradientCheckResult check = Gradients.check(model, new double[] {0.1, -0.2, 0.3}, 2e-5, 2e-5);
		assertTrue(check.message(), check.passed());
	}

	@Test public void dataQualificationUsesSourceProvenanceAndRecursionIsGuarded() {
		String recursive = "functions { real repeat(real x, int n) { "
				+ "if (n == 0) return x; return repeat(x,n-1); } } "
				+ "parameters { real x; } model { target += repeat(x,4); x ~ normal(0,1); }";
		BayesianModel recursiveModel = ModelScript.compileStan(recursive).model();
		assertTrue(Gradients.check(recursiveModel, new double[] {0.2}, 2e-6, 2e-6).passed());

		String invalid = "functions { real trace_data(data matrix A) { return sum(A); } } "
				+ "parameters { matrix[2,2] A; } model { target += trace_data(A * 0); }";
		try { ModelScript.compileStan(invalid); fail("data-qualification diagnostic expected"); }
		catch (ModelScriptException expected) {
			assertTrue(expected.getMessage().contains("no matching overload"));
		}
	}

	@Test public void broadcastCompatibilityMatrixRejectsVectorRowVectorArithmetic() {
		String incompatible = "data { vector[3] x; row_vector[3] y; } parameters { real z; } "
				+ "model { target += sum(x + y) + normal_lpdf(z | 0,1); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("x", new double[] {1, 2, 3}); data.put("y", new double[] {1, 2, 3});
		try { ModelScript.compileStan(incompatible, data); fail("shape diagnostic expected"); }
		catch (ModelScriptException expected) {
			assertTrue(expected.getMessage().contains("incompatible container shapes"));
		}
		String compatible = "data { vector[3] x; array[3] real y; } parameters { real z; } "
				+ "model { x ~ normal(y,z+2); target += sum(exp(x + z)); }";
		assertTrue(Double.isFinite(ModelScript.compileStan(compatible, data).model().logDensity(new double[] {0})));
	}

	@Test public void newConstrainedTypesCompileAndExposeCorrectDimensions() {
		String source = "parameters { unit_vector[3] u; cov_matrix[3] S; corr_matrix[3] R; "
				+ "cholesky_factor_cov[4,3] L; cholesky_factor_corr[3] C; "
				+ "positive_ordered[3] p; } model { target += 0 * (sum(u)+sum(S)+sum(R)+sum(L)+sum(C)+sum(p)); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertEquals(3 + 6 + 3 + 9 + 3 + 3, model.dimension());
		assertTrue(Double.isFinite(model.logDensity(model.initialState())));
	}

	@Test public void literalsForwardDeclarationsAndContainerFunctionsWorkTogether() {
		String source = "functions { real penalty(real x); real penalty(real x) { return square(x); } } "
				+ "transformed data { vector[3] v = [1,2,3]'; row_vector[3] r = [4,5,6]; "
				+ "matrix[2,2] A = [[2,0],[0,3]]; array[2] real a = {7,8}; "
				+ "matrix[3,2] B = append_col(v,v); matrix[2,3] C = append_row(r,r); "
				+ "row_vector[2] cd = columns_dot_product(A,A); vector[2] rd = rows_dot_product(A,A); "
				+ "} parameters { real theta; } model { target += 0 * (sum(B)+sum(C)+sum(cd)+sum(rd)"
				+ "+sum(a)+trace(A)+sum(rep_matrix(v,2))); target += -penalty(theta); theta ~ normal(0,1); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		double[] gradient = new double[1];
		model.logDensityAndGradient(new double[] {.25}, gradient);
		assertEquals(-.75, gradient[0], 1e-12);
		assertTrue(Gradients.check(model, new double[] {.25}, 2e-6, 2e-6).passed());
	}

	@Test public void unresolvedFunctionPrototypeIsACompileError() {
		try {
			ModelScript.compileStan("functions { real missing(real x); } parameters { real x; } model { x ~ normal(0,1); }");
			fail("unresolved function prototype diagnostic expected");
		} catch (ModelScriptException expected) {
			assertTrue(expected.getMessage().contains("has no definition"));
		}
	}
}
