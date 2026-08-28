/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ModelScript;
import jdistlib.matrix.CsrMatrix;

public class StanSparseMatrixTest {
	@Test public void immutableCsrRoundTripsAndMultiplies() {
		double[] dense = {1,0,2, 0,3,0};
		CsrMatrix matrix = CsrMatrix.fromDense(2, 3, dense, 0);
		assertArrayEquals(dense, matrix.toDense(), 0);
		assertArrayEquals(new double[] {7,6}, matrix.multiply(new double[] {1,2,3}), 0);
		assertArrayEquals(new int[] {1,3,4}, matrix.rowStarts());
	}

	@Test public void stanCsrFunctionsDifferentiateWeightsAndVector() {
		String source = "parameters { vector[3] b; vector[3] w; } transformed parameters { "
				+ "vector[2] y = csr_matrix_times_vector(2,3,w,{1,3,2},{1,3,4},b); "
				+ "matrix[2,3] A = csr_to_dense_matrix(2,3,w,{1,3,2},{1,3,4}); } "
				+ "model { b ~ normal(0,1); w ~ normal(0,1); target += sum(y)+0*sum(A); }";
		assertTrue(Gradients.check(ModelScript.compileStan(source).model(),
				new double[] {.1,.2,.3,.4,.5,.6}, 2e-5, 2e-5).passed());
	}

	@Test public void stanCsrExtractionReconstructsDenseMatrix() {
		String source = "transformed data { matrix[2,3] A = [[1,0,2],[0,3,0]]; "
				+ "vector[3] w = csr_extract_w(A); array[3] int v = csr_extract_v(A); "
				+ "array[3] int u = csr_extract_u(A); matrix[2,3] B = csr_to_dense_matrix(2,3,w,v,u); } "
				+ "parameters { real x; } model { target += 0*sum(B); x ~ normal(0,1); }";
		assertTrue(Double.isFinite(ModelScript.compileStan(source).model().logDensity(new double[] {0})));
	}
}
