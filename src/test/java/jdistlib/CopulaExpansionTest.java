/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

public class CopulaExpansionTest {
	@Test
	public void discreteRectangleMassesAreExactAndNormalize() {
		MixedCopulaDistribution independent = new MixedCopulaDistribution(
				new IndependenceCopula(2),
				CopulaMarginal.discrete(new Binomial(1, 0.3)),
				CopulaMarginal.discrete(new Binomial(1, 0.6)));
		assertEquals(0.28, independent.measure(new double[] {0, 0}).value, 2e-16);
		assertEquals(0.12, independent.measure(new double[] {1, 0}).value, 2e-16);
		assertEquals(0.42, independent.measure(new double[] {0, 1}).value, 2e-16);
		assertEquals(0.18, independent.measure(new double[] {1, 1}).value, 2e-16);

		MixedCopulaDistribution dependent = new MixedCopulaDistribution(
				new ClaytonCopula(2, 1.5),
				CopulaMarginal.discrete(new Binomial(1, 0.3)),
				CopulaMarginal.discrete(new Binomial(1, 0.6)));
		double sum = 0.0;
		for (int first = 0; first <= 1; first++) {
			for (int second = 0; second <= 1; second++) {
				CopulaMeasureResult result = dependent.measure(
						new double[] {first, second});
				assertTrue(result.isSuccess());
				assertTrue(result.value >= 0.0);
				sum += result.value;
			}
		}
		assertEquals(1.0, sum, 2e-15);
	}

	@Test
	public void mixedLikelihoodUsesContinuousDerivativeAndDiscreteJump() {
		MixedCopulaDistribution distribution = new MixedCopulaDistribution(
				new IndependenceCopula(2),
				CopulaMarginal.continuous(new Normal(0.0, 1.0)),
				CopulaMarginal.discrete(new Binomial(1, 0.35)));
		double[] observation = {0.4, 1.0};
		CopulaMeasureResult result = distribution.measure(observation);
		assertTrue(result.isSuccess());
		assertEquals(Normal.density(0.4, 0.0, 1.0, false) * 0.35,
				result.value, 2e-9);
		assertTrue(result.absoluteError < 1e-8);
		assertEquals(CopulaMeasureResult.Status.NUMERICAL_MIXED_DERIVATIVE,
				result.getStatus());
		double[] draw = distribution.random(42L);
		assertTrue(Double.isFinite(draw[0]));
		assertTrue(draw[1] == 0.0 || draw[1] == 1.0);
	}

	@Test
	public void pairConditionalCdfsInvertAcrossFamilies() {
		Copula[] copulas = {
				new GaussianCopula(new double[][] {{1.0, 0.55}, {0.55, 1.0}}),
				new StudentTCopula(new double[][] {{1.0, -0.35}, {-0.35, 1.0}}, 5.0),
				new ClaytonCopula(2, 1.3), new GumbelCopula(2, 1.6),
				new FrankCopula(2, -2.5)};
		for (Copula copula : copulas) {
			PairCopula pair = new PairCopula(copula);
			double conditional = pair.conditionalSecondGivenFirst(0.37, 0.68);
			assertTrue(conditional > 0.0 && conditional < 1.0);
			assertEquals(copula.getClass().getSimpleName(), 0.68,
					pair.inverseSecondGivenFirst(0.37, conditional), 2e-6);
		}
	}

	@Test
	public void cVineDensitySamplingAndProbabilityRespectItsConstruction() {
		PairCopula independence = new PairCopula(new IndependenceCopula(2));
		CVineCopula independent = new CVineCopula(
				new PairCopula[] {independence, independence},
				new PairCopula[] {independence});
		double[] point = {0.3, 0.6, 0.8};
		assertEquals(1.0, independent.density(point), 0.0);
		VineProbabilityResult probability = independent.cumulativeResult(point,
				30000, new MersenneTwister(8L));
		assertEquals(0.3 * 0.6 * 0.8, probability.probability,
				4.0 * probability.standardError);

		CVineCopula dependent = new CVineCopula(
				new PairCopula[] {
						new PairCopula(new GaussianCopula(
								new double[][] {{1.0, 0.55}, {0.55, 1.0}})),
						new PairCopula(new ClaytonCopula(2, 1.0))},
				new PairCopula[] {new PairCopula(new FrankCopula(2, 2.0))});
		double[][] sample = dependent.random(1200, 123L);
		assertEquals(dependent.getPairCopula(0, 1).getCopula().kendallsTau(0, 1),
				empiricalTau(sample, 0, 1), 0.06);
		assertEquals(dependent.getPairCopula(0, 2).getCopula().kendallsTau(0, 1),
				empiricalTau(sample, 0, 2), 0.06);
		assertTrue(Double.isFinite(dependent.logDensity(point)));
	}

	@Test
	public void dVineRecursionsPreservePairTreesAndUniformMargins() {
		DVineCopula vine = new DVineCopula(
				new PairCopula[] {
						new PairCopula(new GaussianCopula(
								new double[][] {{1.0, 0.5}, {0.5, 1.0}})),
						new PairCopula(new FrankCopula(2, -2.0))},
				new PairCopula[] {new PairCopula(new ClaytonCopula(2, 0.8))});
		double[][] sample = vine.random(1200, 321L);
		assertEquals(vine.getPairCopula(0, 0).getCopula().kendallsTau(0, 1),
				empiricalTau(sample, 0, 1), 0.06);
		assertEquals(vine.getPairCopula(0, 1).getCopula().kendallsTau(0, 1),
				empiricalTau(sample, 1, 2), 0.06);
		for (int coordinate = 0; coordinate < 3; coordinate++) {
			double mean = 0.0;
			for (double[] row : sample) mean += row[coordinate] / sample.length;
			assertEquals(0.5, mean, 0.025);
		}
		assertTrue(Double.isFinite(vine.logDensity(new double[] {0.3, 0.6, 0.8})));
	}

	@Test
	public void pseudoObservationsAverageTiesAndRejectInvalidUniforms() {
		double[][] pseudo = CopulaFitter.pseudoObservations(new double[][] {
				{1.0, 4.0}, {1.0, 2.0}, {3.0, 3.0}});
		assertEquals(0.375, pseudo[0][0], 0.0);
		assertEquals(0.375, pseudo[1][0], 0.0);
		assertEquals(0.75, pseudo[2][0], 0.0);
		assertEquals(0.75, pseudo[0][1], 0.0);
		assertFalse(CopulaFitter.fitUniforms(new double[][] {{0.0, 0.5}, {0.2, 0.8}},
				CopulaFamily.GAUSSIAN).isSuccess());
	}

	@Test
	public void mixedMarginalTransformsAreDeterministicOrSeeded() {
		double[][] data = {{0.0, 0.0}, {0.5, 1.0}, {-0.5, 1.0}};
		CopulaMarginal[] marginals = {
				CopulaMarginal.continuous(new Normal()),
				CopulaMarginal.discrete(new Binomial(1, 0.35))};
		double[][] midpoint = CopulaFitter.marginalTransforms(data, marginals);
		assertEquals(0.5, midpoint[0][0], 0.0);
		assertEquals(0.325, midpoint[0][1], 2e-16);
		assertEquals(0.825, midpoint[1][1], 2e-16);
		double[][] first = CopulaFitter.marginalTransforms(data,
				new MersenneTwister(4L), marginals);
		double[][] second = CopulaFitter.marginalTransforms(data,
				new MersenneTwister(4L), marginals);
		for (int i = 0; i < data.length; i++) {
			assertEquals(first[i][0], second[i][0], 0.0);
			assertEquals(first[i][1], second[i][1], 0.0);
			assertTrue(first[i][1] > 0.0 && first[i][1] < 1.0);
		}
		assertTrue(CopulaFitter.fitMixed(data, marginals,
				CopulaFamily.INDEPENDENCE).isSuccess());
	}

	@Test
	public void fittingRecoversDependenceAndSelectionFindsClayton() {
		ClaytonCopula source = new ClaytonCopula(2, 2.2);
		double[][] sample = source.random(900, 20260827L);
		CopulaFitResult fit = CopulaFitter.fitUniforms(sample, CopulaFamily.CLAYTON);
		assertTrue(fit.message(), fit.isSuccess());
		assertTrue(fit.getCopula() instanceof ClaytonCopula);
		assertEquals(2.2, ((ClaytonCopula) fit.getCopula()).getTheta(), 0.35);
		assertTrue(Double.isFinite(fit.aic()));
		assertTrue(Double.isFinite(fit.bic()));

		CopulaSelectionResult selection = CopulaSelector.selectUniforms(sample,
				new CopulaFitOptions(), CopulaSelectionCriterion.BIC,
				CopulaFamily.INDEPENDENCE, CopulaFamily.GAUSSIAN,
				CopulaFamily.CLAYTON, CopulaFamily.GUMBEL, CopulaFamily.FRANK);
		assertTrue(selection.isSuccess());
		assertNotNull(selection.getSelected());
		assertEquals(CopulaFamily.CLAYTON, selection.getSelected().getFamily());
		assertEquals(5, selection.getRankings().size());
		assertThrows(UnsupportedOperationException.class,
				() -> selection.getRankings().clear());
	}

	@Test
	public void multivariateGaussianAndStudentFitsHaveValidCorrelation() {
		double[][] sourceCorrelation = {
				{1.0, 0.5, -0.2}, {0.5, 1.0, 0.25}, {-0.2, 0.25, 1.0}};
		double[][] sample = new GaussianCopula(sourceCorrelation).random(700, 77L);
		CopulaFitResult gaussian = CopulaFitter.fitUniforms(sample,
				CopulaFamily.GAUSSIAN);
		CopulaFitResult student = CopulaFitter.fitUniforms(sample,
				CopulaFamily.STUDENT_T,
				new CopulaFitOptions().withOptimizationIterations(20));
		assertTrue(gaussian.message(), gaussian.isSuccess());
		assertTrue(student.message(), student.isSuccess());
		double[][] fitted = ((GaussianCopula) gaussian.getCopula()).getCorrelation();
		assertEquals(0.5, fitted[0][1], 0.08);
		assertEquals(-0.2, fitted[0][2], 0.08);
		assertEquals(0.25, fitted[1][2], 0.08);
	}

	@Test
	public void sequentialVineFittingSelectsEveryPair() {
		CVineCopula source = new CVineCopula(
				new PairCopula[] {
						new PairCopula(new GaussianCopula(
								new double[][] {{1.0, 0.6}, {0.6, 1.0}})),
						new PairCopula(new ClaytonCopula(2, 1.2))},
				new PairCopula[] {new PairCopula(new FrankCopula(2, 2.0))});
		double[][] sample = source.random(550, 818L);
		CopulaFitOptions rankFit = new CopulaFitOptions()
				.withMethod(CopulaFitOptions.Method.KENDALL_TAU);
		VineFitResult cFit = VineFitter.fitUniforms(sample, VineStructure.C_VINE,
				rankFit, CopulaSelectionCriterion.BIC,
				CopulaFamily.INDEPENDENCE, CopulaFamily.GAUSSIAN,
				CopulaFamily.CLAYTON, CopulaFamily.FRANK);
		VineFitResult dFit = VineFitter.fitUniforms(sample, VineStructure.D_VINE,
				rankFit, CopulaSelectionCriterion.BIC,
				CopulaFamily.INDEPENDENCE, CopulaFamily.GAUSSIAN,
				CopulaFamily.CLAYTON, CopulaFamily.FRANK);
		assertTrue(cFit.message(), cFit.isSuccess());
		assertTrue(dFit.message(), dFit.isSuccess());
		assertEquals(3, cFit.getPairSelections().size());
		assertEquals(3, dFit.getPairSelections().size());
		assertTrue(Double.isFinite(cFit.getLogLikelihood()));
		assertTrue(Double.isFinite(dFit.getLogLikelihood()));
		assertTrue(cFit.getParameters() > 0);
		assertTrue(dFit.getParameters() > 0);
	}

	private static double empiricalTau(double[][] sample, int first, int second) {
		long concordant = 0;
		long discordant = 0;
		for (int i = 0; i < sample.length; i++) {
			for (int j = i + 1; j < sample.length; j++) {
				double product = (sample[i][first] - sample[j][first])
						* (sample[i][second] - sample[j][second]);
				if (product > 0.0) concordant++;
				else if (product < 0.0) discordant++;
			}
		}
		return (double) (concordant - discordant) / (concordant + discordant);
	}
}
