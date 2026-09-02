/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;

import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

public class SparseCholeskyPlanTest {
	@Test public void naturalPlanRetainsSparseFillAndScattersBothPrecisions() {
		CsrMatrix matrix = new CsrMatrix(3, 3, new double[] {4,1,3,1,3},
				new int[] {1,1,2,1,3}, new int[] {1,2,4,6});
		SparseCholeskyPlan plan = SparseCholeskyPlan.analyze(matrix,
				MatrixTriangle.LOWER, SparseOrdering.NATURAL);
		assertEquals(3, plan.dimension()); assertEquals(5, plan.structuralNonzeroCount());
		assertEquals(6, plan.factorNonzeroCount());
		assertArrayEquals(new int[] {0,0,1,0,1,2}, plan.factorColumnIndices());
		assertArrayEquals(new int[] {0,1,3,6}, plan.factorRowStarts());
		assertArrayEquals(new int[] {0,1,2}, plan.permutation());
		assertArrayEquals(new double[] {4,1,3,1,0,3}, plan.factorValues(matrix), 0.0);

		FloatCsrMatrix floats = new FloatCsrMatrix(3, 3, new float[] {4,1,3,1,3},
				new int[] {1,1,2,1,3}, new int[] {1,2,4,6});
		SparseCholeskyPlan floatPlan = SparseCholeskyPlan.analyze(floats,
				MatrixTriangle.LOWER, SparseOrdering.NATURAL);
		assertArrayEquals(new float[] {4,1,3,1,0,3}, floatPlan.factorValues(floats), 0.0f);
	}

	@Test public void planRejectsARefactorWithDifferentAuthoritativeStructure() {
		CsrMatrix matrix = new CsrMatrix(2,2,new double[] {4,1,3},
				new int[] {1,1,2},new int[] {1,2,4});
		SparseCholeskyPlan plan = SparseCholeskyPlan.analyze(matrix,
				MatrixTriangle.LOWER,SparseOrdering.MINIMUM_DEGREE);
		try {
			plan.factorValues(new CsrMatrix(2,2,new double[] {4,3},
					new int[] {1,2},new int[] {1,2,3}));
			fail("changed sparse structure should be rejected");
		} catch (IllegalArgumentException expected) {
			assertEquals("sparse refactorization structure differs from analysis", expected.getMessage());
		}
	}
}
