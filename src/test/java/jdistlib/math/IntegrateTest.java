/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.math;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import jdistlib.Normal;

public class IntegrateTest {
	@Test
	public void finiteIntervals() {
		IntegrationResult polynomial = Integrate.integrate(x -> x * x, 0, 1,
				1e-12, 1e-12, 100);
		assertTrue(polynomial.message(), polynomial.isSuccess());
		assertEquals(1.0 / 3.0, polynomial.result, 1e-12);

		IntegrationResult reversed = Integrate.integrate(x -> x * x, 1, 0,
				1e-12, 1e-12, 100);
		assertTrue(reversed.message(), reversed.isSuccess());
		assertEquals(-1.0 / 3.0, reversed.result, 1e-12);

		IntegrationResult endpointSingularity = Integrate.integrate(x -> 1.0 / Math.sqrt(x),
				0, 1, 1e-9, 1e-9, 200);
		assertTrue(endpointSingularity.message(), endpointSingularity.ier <= 2);
		assertEquals(2.0, endpointSingularity.result, 1e-8);
	}

	@Test
	public void infiniteIntervals() {
		IntegrationResult exponential = Integrate.integrate(x -> Math.exp(-x), 0,
				Double.POSITIVE_INFINITY, 1e-10, 1e-10, 200);
		assertTrue(exponential.message(), exponential.isSuccess());
		assertEquals(1.0, exponential.result, 1e-9);

		IntegrationResult normal = Integrate.integrate(
				x -> Normal.density(x, 0, 1, false),
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
				1e-10, 1e-10, 200);
		assertTrue(normal.message(), normal.isSuccess());
		assertEquals(1.0, normal.result, 1e-9);
	}

	@Test
	public void concurrentCallsDoNotShareWorkState() throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(8);
		try {
			List<Callable<Double>> calls = new ArrayList<Callable<Double>>();
			for (int i = 1; i <= 32; i++) {
				final int power = i;
				calls.add(() -> Integrate.integrate(x -> Math.pow(x, power), 0, 1,
						1e-11, 1e-11, 100).result);
			}
			List<Future<Double>> answers = pool.invokeAll(calls);
			for (int i = 0; i < answers.size(); i++)
				assertEquals(1.0 / (i + 2.0), answers.get(i).get(), 1e-10);
		} finally {
			pool.shutdownNow();
		}
	}
}
