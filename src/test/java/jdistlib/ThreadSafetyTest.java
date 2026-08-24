/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class ThreadSafetyTest {
    private static final int SAMPLE_SIZE = 256;

    @Test
    public void cachedRandomGeneratorsHavePerStreamState() throws Exception {
        final double[] expected = sample(20260824L);
        ExecutorService executor = Executors.newFixedThreadPool(8);
        try {
            List<Callable<double[]>> calls = new ArrayList<Callable<double[]>>();
            for (int i = 0; i < 32; i++) {
                calls.add(() -> sample(20260824L));
            }
            for (Future<double[]> future : executor.invokeAll(calls)) {
                assertArrayEquals(expected, future.get(), 0.0);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private static double[] sample(long seed) {
        MersenneTwister random = new MersenneTwister(seed);
        Binomial.RandomState binomial = Binomial.create_random_state();
        Poisson.RandomState poisson = Poisson.create_random_state();
        HyperGeometric.RandomState hypergeometric = HyperGeometric.create_random_state();
        double[] values = new double[SAMPLE_SIZE * 3];
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            values[3 * i] = Binomial.random(250, 0.37, random, binomial);
            values[3 * i + 1] = Poisson.random(42.5, random, poisson);
            values[3 * i + 2] = HyperGeometric.random(120, 180, 75, random,
                    hypergeometric);
        }
        return values;
    }
}
