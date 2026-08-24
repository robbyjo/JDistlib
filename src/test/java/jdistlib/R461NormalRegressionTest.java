/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

/** Regression vectors generated with R 4.6.1. */
public class R461NormalRegressionTest {
    @Test
    public void cumulativePreservesSubnormalTails() {
        assertSubnormalTail(Double.MIN_VALUE,
                Normal.cumulative(-38.46, 0., 1., true, false));
        assertSubnormalTail(0x1.ap-1071,
                Normal.cumulative(-38.4, 0., 1., true, false));
        assertSubnormalTail(0x1.bd91dcp-1049,
                Normal.cumulative(-38., 0., 1., true, false));
        assertSubnormalTail(0x1.8bb5ebc4c36p-1027,
                Normal.cumulative(-37.6, 0., 1., true, false));

        assertSubnormalTail(Double.MIN_VALUE,
                Normal.cumulative(38.46, 0., 1., false, false));
        assertSubnormalTail(0x1.ap-1071,
                Normal.cumulative(38.4, 0., 1., false, false));
        assertSubnormalTail(0x1.bd91dcp-1049,
                Normal.cumulative(38., 0., 1., false, false));
        assertSubnormalTail(0x1.8bb5ebc4c36p-1027,
                Normal.cumulative(37.6, 0., 1., false, false));
    }

    @Test
    public void cumulativeMatchesExtremeLogTailVectors() {
        assertLogTail(-744.15503218861841,
                Normal.cumulative(-38.46, 0., 1., true, true));
        assertLogTail(-741.84767301524835,
                Normal.cumulative(-38.4, 0., 1., true, true));
        assertLogTail(-726.55721601882010,
                Normal.cumulative(-38., 0., 1., true, true));
        assertLogTail(-711.42664867077633,
                Normal.cumulative(-37.6, 0., 1., true, true));
    }

    @Test
    public void infiniteScaleAndInvalidParametersFollowR() {
        assertEquals(0., Normal.cumulative(Double.NEGATIVE_INFINITY,
                3., Double.POSITIVE_INFINITY, true, false), 0.);
        assertEquals(0.5, Normal.cumulative(0., 3.,
                Double.POSITIVE_INFINITY, true, false), 0.);
        assertEquals(1., Normal.cumulative(Double.POSITIVE_INFINITY,
                3., Double.POSITIVE_INFINITY, true, false), 0.);

        assertTrue(Double.isNaN(Normal.density(0., 0.,
                Double.NEGATIVE_INFINITY, false)));
        assertTrue(Double.isNaN(Normal.random(0., -1.,
                new MersenneTwister(1L))));
        assertTrue(Double.isNaN(Normal.random(0.,
                Double.POSITIVE_INFINITY, new MersenneTwister(1L))));
        assertEquals(3., Normal.random(3., 0.,
                new MersenneTwister(1L)), 0.);
    }

    private static void assertSubnormalTail(double expected, double actual) {
        assertTrue("tail probability must not be flushed to zero", actual > 0.);
        assertEquals(expected, actual, 16. * Double.MIN_VALUE);
    }

    private static void assertLogTail(double expected, double actual) {
        assertEquals(expected, actual, 4. * Math.ulp(expected));
    }
}
