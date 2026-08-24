/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.math;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class R461ArithmeticRegressionTest {
    @Test
    public void ldexpHandlesIeee754Boundaries() {
        assertEquals(0., MathFunctions.ldexp(0., -1), 0.);
        assertEquals(Double.doubleToRawLongBits(-0.),
                Double.doubleToRawLongBits(MathFunctions.ldexp(-0., 100)));
        assertEquals(Double.MIN_VALUE, MathFunctions.ldexp(1., -1074), 0.);
        assertEquals(1., MathFunctions.ldexp(Double.MIN_VALUE, 1074), 0.);
        assertEquals(Double.POSITIVE_INFINITY,
                MathFunctions.ldexp(Double.POSITIVE_INFINITY, -1), 0.);
        assertTrue(Double.isNaN(MathFunctions.ldexp(Double.NaN, 10)));
    }

    @Test
    public void frexpReturnsImmediatelyForNonFiniteAndZeroValues() {
        int[] exponent = new int[] { 17 };
        assertEquals(0., MathFunctions.frexp(0., exponent), 0.);
        assertEquals(0, exponent[0]);
        assertEquals(Double.POSITIVE_INFINITY,
                MathFunctions.frexp(Double.POSITIVE_INFINITY, exponent), 0.);
        assertEquals(0, exponent[0]);
        assertTrue(Double.isNaN(MathFunctions.frexp(Double.NaN, exponent)));
        assertEquals(0, exponent[0]);
    }
}
