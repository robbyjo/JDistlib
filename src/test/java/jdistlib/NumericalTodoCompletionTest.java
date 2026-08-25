/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.Test;

import jdistlib.math.CallbackProfile;
import jdistlib.math.ImmutableIntegrationResult;
import jdistlib.math.Integrate;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.IntegrationResult;
import jdistlib.math.IntegrationStatus;
import jdistlib.math.UnivariateFunction;

public class NumericalTodoCompletionTest {
	@Test
	public void immutableResultSnapshotsStatusAndCallbackProfile() {
		IntegrationResult legacy = Integrate.integrate(x -> x * x, 0.0, 1.0,
				IntegrationOptions.builder().tolerances(1e-12, 1e-12).build());
		ImmutableIntegrationResult result = legacy.toImmutable();
		legacy.result = -99.0;

		assertTrue(result.isSuccess());
		assertEquals(IntegrationStatus.SUCCESS, result.getStatus());
		assertEquals(1.0 / 3.0, result.getValue(), 2e-14);
		assertTrue(result.getEvaluationCount() > 0);
		CallbackProfile profile = result.getCallbackProfile();
		assertEquals(result.getEvaluationCount(), profile.getAttemptedEvaluations());
		assertEquals(profile.getAttemptedEvaluations(),
				profile.getCompletedEvaluations());
		assertTrue(profile.getTotalCallbackNanos() >= 0L);
		assertTrue(profile.getMaximumCallbackNanos() >= 0L);
		assertTrue(profile.getIntegrationWallNanos() >=
				profile.getMaximumCallbackNanos());
	}

	@Test
	public void callbackTimeLimitsAndIsolationFailClosed() {
		IntegrationOptions observedLimit = IntegrationOptions.builder()
				.maxCallbackTime(1L, TimeUnit.NANOSECONDS).build();
		ImmutableIntegrationResult observed = Integrate.integrateImmutable(
				x -> x, 0.0, 1.0, observedLimit);
		assertEquals(IntegrationStatus.CALLBACK_TIME_LIMIT_EXCEEDED,
				observed.getStatus());

		IntegrationOptions isolated = IntegrationOptions.builder()
				.callbackExecution(IntegrationOptions.CallbackExecution.ISOLATED_DAEMON)
				.maxCallbackTime(10L, TimeUnit.MILLISECONDS)
				.maxTotalTime(100L, TimeUnit.MILLISECONDS)
				.build();
		long start = System.nanoTime();
		ImmutableIntegrationResult timedOut = Integrate.integrateImmutable(x -> {
			try {
				Thread.sleep(10000L);
			} catch (InterruptedException exception) {
				Thread.currentThread().interrupt();
			}
			return x;
		}, 0.0, 1.0, isolated);
		long elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
		assertEquals(IntegrationStatus.CALLBACK_TIME_LIMIT_EXCEEDED,
				timedOut.getStatus());
		assertTrue("isolated timeout took " + elapsedMillis + " ms",
				elapsedMillis < 1000L);
		assertEquals(1, timedOut.getCallbackProfile().getAttemptedEvaluations());
		assertEquals(0, timedOut.getCallbackProfile().getCompletedEvaluations());

		ImmutableIntegrationResult totalLimit = Integrate.integrateImmutable(
				x -> x, 0.0, 1.0, IntegrationOptions.builder()
						.maxTotalTime(1L, TimeUnit.NANOSECONDS).build());
		assertEquals(IntegrationStatus.CALLBACK_TIME_LIMIT_EXCEEDED,
				totalLimit.getStatus());

		assertThrows(IllegalArgumentException.class, () ->
				IntegrationOptions.builder().callbackExecution(
						IntegrationOptions.CallbackExecution.ISOLATED_DAEMON).build());
	}

	@Test
	public void analyzerDirectProbesHonorWorkerIsolation() {
		IntegrationOptions isolated = IntegrationOptions.builder()
				.callbackExecution(IntegrationOptions.CallbackExecution.ISOLATED_DAEMON)
				.maxCallbackTime(5L, TimeUnit.MILLISECONDS)
				.maxTotalTime(100L, TimeUnit.MILLISECONDS)
				.build();
		long start = System.nanoTime();
		FunctionAnalysis analysis = ProbabilityFunctionAnalyzer.analyze(x -> {
			try {
				Thread.sleep(10000L);
			} catch (InterruptedException exception) {
				Thread.currentThread().interrupt();
			}
			return 1.0;
		}, 0.0, 1.0, FunctionAnalysisOptions.builder()
				.sampleCount(9).randomizedProbeBudget(0).repeatabilityChecks(0)
				.integrationOptions(isolated).build());
		long elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
		assertTrue(hasFinding(analysis, "CALLBACK_TIME_LIMIT"));
		assertTrue("isolated analysis took " + elapsedMillis + " ms",
				elapsedMillis < 1000L);
	}

	@Test
	public void diagnosticReportsSerializeAsVersionedJson() {
		DiagnosticFinding finding = new DiagnosticFinding(
				DiagnosticFinding.Severity.WARNING, "QUOTED",
				"line one\n\"line two\"", Double.NaN);
		String findingJson = finding.toJson();
		assertTrue(findingJson.contains("\"severity\":\"WARNING\""));
		assertTrue(findingJson.contains("line one\\n\\\"line two\\\""));
		assertTrue(findingJson.contains("\"x\":null"));

		FunctionAnalysis function = ProbabilityFunctionAnalyzer.analyze(
				x -> 1.0, 0.0, 1.0,
				FunctionAnalysisOptions.builder().sampleCount(9)
						.randomizedProbeBudget(0).build());
		String functionJson = function.toJson();
		assertTrue(functionJson.startsWith("{\"schemaVersion\":1"));
		assertTrue(functionJson.contains("\"type\":\"functionAnalysis\""));
		assertTrue(functionJson.contains("\"normalizationStability\":"));

		DistributionAnalysis distribution = new NumericalDiscreteDistribution(
				x -> x + 1.0, 0, 2).analyzeDistribution();
		String distributionJson = distribution.toJson();
		assertTrue(distributionJson.contains("\"type\":\"distributionAnalysis\""));
		assertTrue(distributionJson.contains("\"absoluteMoments\":"));
		assertFalse(distributionJson.contains("NaN"));

		String integrationJson = Integrate.integrateImmutable(x -> x, 0.0, 1.0)
				.toJson();
		assertTrue(integrationJson.contains("\"callbackProfile\":"));
		assertTrue(integrationJson.endsWith("}"));
	}

	@Test
	public void highPrecisionReferenceCorpusCoversDifficultFamilies()
			throws IOException {
		List<ReferenceCase> cases = loadReferenceCases();
		assertEquals(6, cases.size());
		for (ReferenceCase reference : cases) {
			IntegrationOptions.Builder builder = IntegrationOptions.builder()
					.tolerances(0.0, reference.relativeTolerance)
					.subdivisions(2000)
					.maxEvaluations(1000000)
					.method(reference.method)
					.tanhSinhMaxLevels(16);
			if (reference.breakpoints.length > 0) {
				builder.breakpoints(reference.breakpoints);
			}
			IntegrationResult actual = Integrate.integrate(function(reference.id),
					reference.lower, reference.upper, builder.build());
			assertTrue(reference.id + ": " + actual.detailedMessage(),
					actual.isSuccess());
			double relativeError = Math.abs(actual.result - reference.expected)
					/ Math.abs(reference.expected);
			assertTrue(reference.id + " relative error=" + relativeError,
					relativeError <= reference.relativeTolerance);
		}
	}

	private static UnivariateFunction function(String id) {
		if ("oscillatory_sine_squared".equals(id)) {
			return x -> {
				double sine = Math.sin(100.0 * x);
				return sine * sine;
			};
		}
		if ("endpoint_beta_half".equals(id)) {
			return x -> 1.0 / Math.sqrt(x * (1.0 - x));
		}
		if ("interior_inverse_square_root".equals(id)) {
			return x -> 1.0 / Math.sqrt(Math.abs(x - 0.3));
		}
		if ("large_scaled_polynomial".equals(id)) {
			return x -> 1e200 * Math.pow(x, 20.0);
		}
		if ("narrow_triangular_mode".equals(id)) {
			return x -> Math.max(0.0, 1.0 - Math.abs(x - 0.123456) / 0.000001);
		}
		if ("pareto_quarter_tail".equals(id)) {
			return x -> Math.pow(1.0 + x, -1.25);
		}
		throw new IllegalArgumentException("unknown reference function " + id);
	}

	private static boolean hasFinding(FunctionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (code.equals(finding.getCode())) return true;
		}
		return false;
	}

	private static List<ReferenceCase> loadReferenceCases() throws IOException {
		InputStream stream = NumericalTodoCompletionTest.class.getResourceAsStream(
				"/jdistlib/math/integration-reference.csv");
		assertNotNull(stream);
		List<ReferenceCase> result = new ArrayList<ReferenceCase>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(
				stream, StandardCharsets.UTF_8))) {
			reader.readLine();
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.length() == 0) continue;
				String[] fields = line.split(",", -1);
				result.add(new ReferenceCase(fields));
			}
		}
		return result;
	}

	private static final class ReferenceCase {
		final String id;
		final double lower;
		final double upper;
		final IntegrationOptions.Method method;
		final double[] breakpoints;
		final double expected;
		final double relativeTolerance;

		ReferenceCase(String[] fields) {
			id = fields[0];
			lower = parseDouble(fields[2]);
			upper = parseDouble(fields[3]);
			method = IntegrationOptions.Method.valueOf(fields[4]);
			breakpoints = parseBreakpoints(fields[5]);
			expected = Double.parseDouble(fields[6]);
			relativeTolerance = Double.parseDouble(fields[7]);
		}

		private static double parseDouble(String value) {
			return "Infinity".equals(value) ? Double.POSITIVE_INFINITY
					: "-Infinity".equals(value) ? Double.NEGATIVE_INFINITY
					: Double.parseDouble(value);
		}

		private static double[] parseBreakpoints(String value) {
			if (value.length() == 0) return new double[0];
			String[] parts = value.split(";");
			double[] result = new double[parts.length];
			for (int i = 0; i < parts.length; i++) {
				result[i] = Double.parseDouble(parts[i]);
			}
			return result;
		}
	}
}
