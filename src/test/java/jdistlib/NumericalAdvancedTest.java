/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.math.IntegrationOptions;

public class NumericalAdvancedTest {
	private static final IntegrationOptions ACCURATE = IntegrationOptions.builder()
			.tolerances(1e-10, 1e-10)
			.subdivisions(400)
			.maxEvaluations(500000)
			.method(IntegrationOptions.Method.AUTO)
			.build();

	@Test
	public void diagnosticsRequireAbsoluteMomentConvergence() {
		NumericalContinuousDistribution cauchy =
				new NumericalContinuousDistribution(x -> 1.0 / (1.0 + x * x),
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, ACCURATE);
		DistributionAnalysis analysis = cauchy.analyzeDistribution();
		assertFalse(analysis.areMomentsStable());
		assertTrue(hasCode(analysis, "ABSOLUTE_MOMENTS_UNSTABLE"));
	}

	@Test
	public void reusableCdfTableIsMonotoneAccurateAndReused() {
		NumericalContinuousDistribution normal =
				new NumericalContinuousDistribution(
						x -> Math.exp(-0.5 * x * x),
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, ACCURATE);
		NumericalCdfTable table = normal.rebuildCdfTable(CdfTableOptions.builder()
				.tolerance(2e-9).initialIntervals(32).maximumNodes(2049).build());
		assertTrue(table.size() >= 33);
		assertSame(table, normal.getCdfTable());
		double previous = 0.0;
		for (double x : new double[] {-4.0, -2.0, -0.5, 0.0, 0.7, 2.0, 4.0}) {
			double cached = normal.cumulativeCached(x, true, false);
			assertTrue(cached >= previous);
			assertEquals(Normal.cumulative(x, 0.0, 1.0, true, false), cached, 2e-8);
			previous = cached;
		}
		for (double p : new double[] {0.01, 0.1, 0.5, 0.9, 0.99}) {
			assertEquals(Normal.quantile(p, 0.0, 1.0, true, false),
					normal.quantile(p, true, false), 3e-8);
		}
	}

	@Test
	public void automaticLogScalingFindsAndCombinesMultipleModes() {
		double rootTwoPi = Math.sqrt(2.0 * Math.PI);
		NumericalContinuousDistribution mixture =
				NumericalContinuousDistribution.fromLogKernel(x -> {
					double narrow = 1000.0 - 0.5 * Math.pow((x + 3.0) / 0.1, 2.0);
					double broad = 995.0 - 0.5 * Math.pow((x - 4.0) / 10.0, 2.0);
					return logAdd(narrow, broad);
				}, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, ACCURATE);
		double expected = logAdd(1000.0 + Math.log(0.1 * rootTwoPi),
				995.0 + Math.log(10.0 * rootTwoPi));
		assertTrue(mixture.getLogScalingRegionCount() >= 2);
		assertEquals(expected, mixture.getLogNormalizationConstant(), 2e-8);
		assertTrue(mixture.cumulative(0.0, true, false) > 0.0);
		assertTrue(mixture.cumulative(0.0, true, false) < 1.0);
	}

	@Test
	public void piecewiseSupportHandlesHolesAndAtomsExactly() {
		NumericalSupport support = NumericalSupport.builder()
				.interval(-2.0, 2.0)
				.hole(-0.5, 0.5)
				.atom(0.0)
				.singularity(1.0)
				.build();
		NumericalPiecewiseDistribution distribution =
				new NumericalPiecewiseDistribution(x -> 1.0, support, x -> 2.0,
						ACCURATE);
		assertEquals(5.0, distribution.getNormalizationConstant(), 2e-12);
		assertEquals(0.2, distribution.density(-1.0, false), 2e-13);
		assertEquals(0.0, distribution.density(0.25, false), 0.0);
		assertEquals(0.4, distribution.density(0.0, false), 2e-13);
		assertEquals(0.3, distribution.cumulative(-0.5, true, false), 2e-12);
		assertEquals(0.7, distribution.cumulative(0.0, true, false), 2e-12);
		assertEquals(0.0, distribution.quantile(0.5, true, false), 0.0);

		NumericalPiecewiseDistribution logged =
				NumericalPiecewiseDistribution.fromLogKernel(x -> 0.0, support,
						x -> Math.log(2.0), ACCURATE);
		assertEquals(Math.log(5.0), logged.getLogNormalizationConstant(), 2e-12);
		assertEquals(0.4, logged.density(0.0, false), 2e-13);
	}

	@Test
	public void certifiedInfiniteDiscreteSupportRequiresAndReportsTailBounds() {
		double ratio = 0.35;
		CertifiedDiscreteOptions options = CertifiedDiscreteOptions.builder()
				.minimumTerms(1)
				.maximumTerms(1000)
				.omittedProbabilityTolerance(1e-10)
				.build();
		CertifiedInfiniteDiscreteDistribution geometric =
				CertifiedInfiniteDiscreteDistribution.rightInfinite(
						k -> Math.pow(ratio, k), 0L,
						(first, firstWeight) -> firstWeight / (1.0 - ratio), options);
		assertTrue(geometric.getIncludedTerms() < 100);
		assertTrue(geometric.getOmittedProbabilityUpperBound() <= 1e-10);
		assertEquals(1.0 - ratio, geometric.density(0.0, false), 1e-10);
		assertEquals(0.0, geometric.quantile(0.5, true, false), 0.0);

		CertifiedInfiniteDiscreteDistribution twoSided =
				CertifiedInfiniteDiscreteDistribution.twoSided(
						k -> Math.pow(ratio, Math.abs(k)), 0L,
						(first, firstWeight) -> firstWeight / (1.0 - ratio),
						(first, firstWeight) -> firstWeight / (1.0 - ratio), options);
		assertTrue(twoSided.getOmittedProbabilityUpperBound() <= 1e-10);
		assertEquals(0.0, twoSided.quantile(0.5, true, false), 0.0);

		assertThrows(IllegalArgumentException.class,
				() -> CertifiedInfiniteDiscreteDistribution.rightInfinite(
						k -> Math.pow(ratio, k), 0L,
						(first, firstWeight) -> firstWeight * 0.5, options));
	}

	private static boolean hasCode(DistributionAnalysis analysis, String code) {
		for (DiagnosticFinding finding : analysis.getFindings()) {
			if (finding.getCode().equals(code)) return true;
		}
		return false;
	}

	private static double logAdd(double x, double y) {
		double high = Math.max(x, y);
		return high + Math.log1p(Math.exp(Math.min(x, y) - high));
	}
}
