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
	public void cquadIntegratesSmoothAndNonsmoothFiniteFunctions() {
		IntegrationOptions options = IntegrationOptions.builder()
				.tolerances(1e-11, 1e-11)
				.subdivisions(200)
				.maxEvaluations(200000)
				.method(IntegrationOptions.Method.CQUAD)
				.build();
		IntegrationResult exponential = Integrate.integrate(Math::exp,
				-1.0, 1.0, options);
		assertTrue(exponential.detailedMessage(), exponential.isSuccess());
		assertEquals(Math.E - 1.0 / Math.E, exponential.result, 2e-13);

		double location = 0.123456789;
		IntegrationResult cusp = Integrate.integrate(
				x -> Math.abs(x - location), -1.0, 1.0, options);
		double expected = 0.5 * ((location + 1.0) * (location + 1.0)
				+ (1.0 - location) * (1.0 - location));
		assertTrue(cusp.detailedMessage(), cusp.isSuccess());
		assertEquals(expected, cusp.result, 2e-10);

		IntegrationResult localizedPeak = Integrate.integrate(
				x -> 1.0 / (1.0 + 10000.0 * (x - 0.2) * (x - 0.2)),
				0.0, 1.0, options);
		double peakExpected = (Math.atan(80.0) + Math.atan(20.0)) / 100.0;
		assertTrue(localizedPeak.detailedMessage(), localizedPeak.isSuccess());
		assertEquals(peakExpected, localizedPeak.result, 2e-11);
	}

	@Test
	public void cquadHonorsBreakpointsBudgetsAndFiniteDomainContract() {
		IntegrationOptions split = IntegrationOptions.builder()
				.tolerances(1e-12, 1e-12)
				.breakpoints(0.25)
				.method(IntegrationOptions.Method.CQUAD)
				.build();
		IntegrationResult discontinuous = Integrate.integrate(
				x -> x < 0.25 ? 1.0 : 3.0, 0.0, 1.0, split);
		assertTrue(discontinuous.detailedMessage(), discontinuous.isSuccess());
		assertEquals(2.5, discontinuous.result, 2e-12);

		IntegrationOptions budget = split.toBuilder().breakpoints()
				.maxEvaluations(5).build();
		IntegrationResult exhausted = Integrate.integrate(Math::exp,
				0.0, 1.0, budget);
		assertEquals(IntegrationStatus.EVALUATION_BUDGET_EXHAUSTED,
				exhausted.getStatus());

		IntegrationResult invalid = Integrate.integrate(x -> Math.exp(-x),
				0.0, Double.POSITIVE_INFINITY, split);
		assertEquals(IntegrationStatus.INVALID_INPUT, invalid.getStatus());
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
