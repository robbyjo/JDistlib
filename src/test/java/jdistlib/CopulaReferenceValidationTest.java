/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class CopulaReferenceValidationTest {
	@Test
	public void mixedMeasuresMatchIndependentHighPrecisionReferences()
			throws IOException {
		for (MixedReference reference : mixedReferences()) {
			MixedCopulaDistribution distribution;
			double[] observation;
			if ("mixed".equals(reference.kind)) {
				distribution = new MixedCopulaDistribution(
						new ClaytonCopula(2, reference.theta),
						CopulaMarginal.continuous(new Uniform(0.0, 1.0)),
						CopulaMarginal.discrete(
								new Binomial(1.0, reference.firstProbability)));
				observation = new double[] {reference.continuous,
						reference.firstOutcome};
			} else {
				distribution = new MixedCopulaDistribution(
						new ClaytonCopula(2, reference.theta),
						CopulaMarginal.discrete(
								new Binomial(1.0, reference.firstProbability)),
						CopulaMarginal.discrete(
								new Binomial(1.0, reference.secondProbability)));
				observation = new double[] {reference.firstOutcome,
						reference.secondOutcome};
			}
			CopulaMeasureResult actual = distribution.measure(observation);
			assertTrue(reference.id + ": " + actual.message(), actual.hasEstimate());
			double tolerance = Math.max(reference.absoluteTolerance,
					reference.relativeTolerance * Math.abs(reference.expected));
			assertEquals(reference.id, reference.expected, actual.value, tolerance);
			if ("discrete".equals(reference.kind)) {
				assertEquals(CopulaMeasureResult.Status.RECTANGLE_DIFFERENCE,
						actual.getStatus());
			}
		}
	}

	@Test
	public void vineLogDensitiesMatchIndependentFactorizations()
			throws IOException {
		for (VineReference reference : vineReferences()) {
			PairCopula first = clayton(reference.theta0);
			PairCopula second = clayton(reference.theta1);
			PairCopula conditional = clayton(reference.theta2);
			VineCopula vine = reference.structure == VineStructure.C_VINE
					? new CVineCopula(new PairCopula[] {first, second},
							new PairCopula[] {conditional})
					: new DVineCopula(new PairCopula[] {first, second},
							new PairCopula[] {conditional});
			assertEquals(reference.id, reference.expected,
					vine.logDensity(reference.point), reference.tolerance);
		}
	}

	@Test
	public void analyticPairConditionalsRemainInvertibleNearBoundaries() {
		PairCopula[] pairs = {
				new PairCopula(new ClaytonCopula(2, 2.3)),
				new PairCopula(new GumbelCopula(2, 1.8)),
				new PairCopula(new FrankCopula(2, -4.0)),
				new PairCopula(new FrankCopula(2, 5.0))};
		double[][] points = {{1e-7, 2e-5}, {1e-3, 0.98},
				{0.999999, 0.03}, {0.9999, 0.99999}};
		for (PairCopula pair : pairs) {
			for (double[] point : points) {
				double conditional = pair.conditionalSecondGivenFirst(
						point[0], point[1]);
				assertTrue(conditional >= 0.0 && conditional <= 1.0);
				assertEquals(pair.getCopula().getClass().getSimpleName(), point[1],
						pair.inverseSecondGivenFirst(point[0], conditional), 2e-8);
			}
		}
	}

	@Test
	public void likelihoodDiagnosticsExposeBoundaryAndRowInformation() {
		Copula copula = new ClaytonCopula(2, 1.4);
		double[][] observations = {
				{0.4, 0.6}, {1e-8, 0.03}, {0.999999, 0.2}};
		CopulaLikelihoodDiagnostics diagnostics =
				CopulaLikelihoodDiagnostics.assess(copula, observations);
		assertTrue(diagnostics.message(), diagnostics.isSuccess());
		assertEquals(3, diagnostics.getObservations());
		assertEquals(3, diagnostics.getFiniteContributions());
		assertEquals(2, diagnostics.countNearBoundary(1.1e-6));
		assertEquals(1e-8, diagnostics.getMinimumBoundaryDistance(), 1e-20);
		assertEquals(diagnostics.getLogLikelihood() / 3.0,
				diagnostics.getMeanLogContribution(), 2e-15);
		double[] firstCopy = diagnostics.getLogContributions();
		double[] secondCopy = diagnostics.getLogContributions();
		assertNotSame(firstCopy, secondCopy);
		assertArrayEquals(firstCopy, secondCopy, 0.0);

		CopulaLikelihoodDiagnostics boundary =
				CopulaLikelihoodDiagnostics.assess(copula,
						new double[][] {{0.4, 0.6}, {0.0, 0.5}});
		assertFalse(boundary.isSuccess());
		assertEquals(CopulaLikelihoodDiagnostics.Status.NONFINITE_CONTRIBUTION,
				boundary.getStatus());
		assertEquals(1, boundary.getFirstProblemIndex());
	}

	@Test
	public void mixedLikelihoodResultRetainsContributionDiagnostics() {
		MixedCopulaDistribution distribution = new MixedCopulaDistribution(
				new ClaytonCopula(2, 1.2),
				CopulaMarginal.continuous(new Normal()),
				CopulaMarginal.discrete(new Binomial(1.0, 0.3)));
		double[][] observations = {{0.1, 0.0}, {-0.4, 1.0}, {1.2, 1.0}};
		CopulaLogLikelihoodResult result =
				distribution.logLikelihoodResult(observations);
		assertTrue(result.message(), result.hasEstimate());
		assertEquals(3, result.getObservations());
		assertEquals(3, result.getSuccessfulContributions());
		assertTrue(result.getCdfEvaluations() > 0);
		assertEquals(distribution.logLikelihood(observations),
				result.getLogLikelihood(), 0.0);
		assertNotSame(result.getContributions(), result.getContributions());

		CopulaLogLikelihoodResult invalid = distribution.logLikelihoodResult(
				new double[][] {{0.1, 0.0}, null});
		assertFalse(invalid.hasEstimate());
		assertEquals(1, invalid.getFirstProblemIndex());
		assertEquals(CopulaLogLikelihoodResult.Status.MEASURE_FAILURE,
				invalid.getStatus());
	}

	@Test
	public void fittedResultsExposeDiagnosticsAndVineInformationCriteria() {
		double[][] sample = new ClaytonCopula(2, 1.6).random(180, 9102L);
		CopulaFitResult pair = CopulaFitter.fitUniforms(sample,
				CopulaFamily.CLAYTON,
				new CopulaFitOptions().withMethod(
						CopulaFitOptions.Method.KENDALL_TAU));
		assertTrue(pair.message(), pair.isSuccess());
		assertTrue(pair.getDiagnostics().isSuccess());
		assertEquals(sample.length, pair.getDiagnostics().getObservations());
		assertEquals(pair.getLogLikelihood(),
				pair.getDiagnostics().getLogLikelihood(), 0.0);

		double[][] vineSample = new CVineCopula(
				new PairCopula[] {clayton(1.0), clayton(0.7)},
				new PairCopula[] {clayton(1.3)}).random(160, 781L);
		VineFitResult vine = VineFitter.fitUniforms(vineSample,
				VineStructure.C_VINE,
				new CopulaFitOptions().withMethod(
						CopulaFitOptions.Method.KENDALL_TAU),
				CopulaSelectionCriterion.BIC, CopulaFamily.INDEPENDENCE,
				CopulaFamily.CLAYTON);
		assertTrue(vine.message(), vine.isSuccess());
		assertEquals(vineSample.length, vine.getObservations());
		assertTrue(vine.getDiagnostics().isSuccess());
		assertEquals(2.0 * vine.getParameters() - 2.0 * vine.getLogLikelihood(),
				vine.aic(), 0.0);
		assertEquals(Math.log(vine.getObservations()) * vine.getParameters()
				- 2.0 * vine.getLogLikelihood(), vine.bic(), 0.0);
	}

	@Test
	public void seedConveniencesMatchCallerOwnedEnginesExactly() {
		double[][] data = {{-0.4, 0.0}, {0.1, 1.0}, {0.7, 1.0}, {1.2, 0.0}};
		CopulaMarginal[] marginals = {
				CopulaMarginal.continuous(new Normal()),
				CopulaMarginal.discrete(new Binomial(1.0, 0.4))};
		double[][] bySeed = CopulaFitter.marginalTransforms(data, 1234L, marginals);
		double[][] byEngine = CopulaFitter.marginalTransforms(data,
				new MersenneTwister(1234L), marginals);
		for (int row = 0; row < data.length; row++)
			assertArrayEquals(byEngine[row], bySeed[row], 0.0);

		CopulaFitOptions options = new CopulaFitOptions().withMethod(
				CopulaFitOptions.Method.KENDALL_TAU);
		CopulaSelectionResult first = CopulaSelector.selectMixed(data, marginals,
				1234L, options, CopulaSelectionCriterion.BIC,
				CopulaFamily.INDEPENDENCE, CopulaFamily.FRANK);
		CopulaSelectionResult second = CopulaSelector.selectMixed(data, marginals,
				new MersenneTwister(1234L), options,
				CopulaSelectionCriterion.BIC, CopulaFamily.INDEPENDENCE,
				CopulaFamily.FRANK);
		assertEquals(first.getSelected().getFamily(),
				second.getSelected().getFamily());
		assertEquals(first.getSelected().getLogLikelihood(),
				second.getSelected().getLogLikelihood(), 0.0);
	}

	private static PairCopula clayton(double theta) {
		return new PairCopula(new ClaytonCopula(2, theta));
	}

	private static List<MixedReference> mixedReferences() throws IOException {
		List<String[]> rows = rows("/jdistlib/copula/mixed-reference.csv");
		List<MixedReference> result = new ArrayList<MixedReference>();
		for (String[] row : rows) result.add(new MixedReference(row));
		return result;
	}

	private static List<VineReference> vineReferences() throws IOException {
		List<String[]> rows = rows("/jdistlib/copula/vine-reference.csv");
		List<VineReference> result = new ArrayList<VineReference>();
		for (String[] row : rows) result.add(new VineReference(row));
		return result;
	}

	private static List<String[]> rows(String resource) throws IOException {
		InputStream stream = CopulaReferenceValidationTest.class
				.getResourceAsStream(resource);
		if (stream == null) throw new IOException("missing resource " + resource);
		List<String[]> result = new ArrayList<String[]>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(
				stream, StandardCharsets.UTF_8))) {
			reader.readLine();
			String line;
			while ((line = reader.readLine()) != null)
				if (line.length() > 0) result.add(line.split(",", -1));
		}
		return result;
	}

	private static final class MixedReference {
		final String id;
		final String kind;
		final double theta;
		final double continuous;
		final double firstProbability;
		final double firstOutcome;
		final double secondProbability;
		final double secondOutcome;
		final double expected;
		final double absoluteTolerance;
		final double relativeTolerance;

		MixedReference(String[] row) {
			id = row[0];
			kind = row[1];
			theta = Double.parseDouble(row[2]);
			continuous = number(row[3]);
			firstProbability = Double.parseDouble(row[4]);
			firstOutcome = Double.parseDouble(row[5]);
			secondProbability = number(row[6]);
			secondOutcome = number(row[7]);
			expected = Double.parseDouble(row[8]);
			absoluteTolerance = Double.parseDouble(row[9]);
			relativeTolerance = Double.parseDouble(row[10]);
		}
	}

	private static final class VineReference {
		final String id;
		final VineStructure structure;
		final double[] point;
		final double theta0;
		final double theta1;
		final double theta2;
		final double expected;
		final double tolerance;

		VineReference(String[] row) {
			id = row[0];
			structure = VineStructure.valueOf(row[1]);
			point = new double[] {Double.parseDouble(row[2]),
					Double.parseDouble(row[3]), Double.parseDouble(row[4])};
			theta0 = Double.parseDouble(row[5]);
			theta1 = Double.parseDouble(row[6]);
			theta2 = Double.parseDouble(row[7]);
			expected = Double.parseDouble(row[8]);
			tolerance = Double.parseDouble(row[9]);
		}
	}

	private static double number(String value) {
		return value.length() == 0 ? Double.NaN : Double.parseDouble(value);
	}
}
