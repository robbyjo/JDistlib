/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.math;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.Poisson;

public class VectorMathTest {
    @Test
    public void scaledEqualityIsSymmetricForNegativeValues() {
        assertTrue(VectorMath.isEqualScaled(-1.0e100, -1.0e100 * (1 + 1e-14), 2e-14));
        assertFalse(VectorMath.isEqualScaled(-1.0e100, -2.0e100, 1e-12));
    }

    @Test
    public void vectorEqualityHandlesScaleAndSpecialValues() {
        assertTrue(VectorMath.allEqual(
                new double[] {-5000, Double.POSITIVE_INFINITY, Double.NaN},
                new double[] {-5000 + 5e-13, Double.POSITIVE_INFINITY, Double.NaN},
                2e-16));
        assertFalse(VectorMath.allEqual(
                new double[] {Double.POSITIVE_INFINITY},
                new double[] {Double.NEGATIVE_INFINITY}, 1e-12));
    }

    @Test
    public void poissonDensityDoesNotNarrowLargeIntegerValuedDoubles() {
        assertTrue(VectorMath.isEqualScaled(
                -7.1280137882815411781632e22,
                Poisson.density(1e20, 1e-290, true), 2e-15));
    }
}
