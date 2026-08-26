/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

import jdistlib.disttest.DiscreteFdr;
import jdistlib.disttest.DiscretePValueDistribution;

public class DiscreteFdrTest {
	@Test
	public void dbhCriticalValuesMatchDirectTransformCalculation() {
		DiscretePValueDistribution[] nulls = {
			DiscretePValueDistribution.exact(
					new double[] {0.01, 0.05, 0.20, 1.0}),
			DiscretePValueDistribution.exact(
					new double[] {0.02, 0.10, 0.50, 1.0})
		};
		double[] pValues = {0.02, 0.05};
		DiscreteFdr.Result down = DiscreteFdr.dbhStepDown(pValues, nulls, 0.05);
		assertArrayEquals(new double[] {0.02, 0.05},
				down.getCriticalValues(), 0.0);
		assertArrayEquals(new boolean[] {true, true}, down.getRejected());
		assertEquals(0.05, down.getThreshold(), 0.0);
		DiscreteFdr.Result up = DiscreteFdr.dbhStepUp(pValues, nulls, 0.05);
		assertArrayEquals(new double[] {0.02, 0.05},
				up.getCriticalValues(), 0.0);
		assertEquals(2, up.getRejectedCount());
	}

	@Test
	public void customDiscreteCdfAndInvalidNullsAreHandled() {
		DiscretePValueDistribution distribution =
				new DiscretePValueDistribution(
						new double[] {0.1, 0.5, 1.0},
						new double[] {0.05, 0.4, 1.0});
		assertEquals(0.0, distribution.cdf(0.09), 0.0);
		assertEquals(0.05, distribution.cdf(0.2), 0.0);
		assertThrows(IllegalArgumentException.class, () ->
				new DiscretePValueDistribution(
						new double[] {0.1, 1.0},
						new double[] {0.2, 1.0}));
		assertThrows(IllegalArgumentException.class, () ->
				DiscreteFdr.dbhStepDown(new double[] {Double.NaN},
						new DiscretePValueDistribution[] {distribution}, 0.05));
	}
}
