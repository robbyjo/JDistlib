/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Copula;

/** Tail dependence, finite-level concentration, and bivariate stress regions. */
public final class CopulaTailAnalysis {
	private CopulaTailAnalysis() {}
	public static NumericalEstimate lowerTailDependence(Copula copula) {
		return coefficient(copula, true);
	}
	public static NumericalEstimate upperTailDependence(Copula copula) {
		return coefficient(copula, false);
	}
	public static double lowerConcentration(Copula copula, double level) {
		require(copula, level); return copula.cumulative(new double[] {level, level}) / level;
	}
	public static double upperConcentration(Copula copula, double level) {
		require(copula, level); return (1.0 - 2.0 * level
				+ copula.cumulative(new double[] {level, level})) / (1.0 - level);
	}
	public static double stressProbability(Copula copula, Tail firstTail, double firstLevel,
			Tail secondTail, double secondLevel) {
		if (copula == null || copula.dimension() != 2 || firstTail == null || secondTail == null
				|| firstLevel < 0.0 || firstLevel > 1.0 || secondLevel < 0.0 || secondLevel > 1.0)
			throw new IllegalArgumentException("bivariate copula, tails, and unit levels required");
		double c = copula.cumulative(new double[] {firstLevel, secondLevel});
		if (firstTail == Tail.LOWER && secondTail == Tail.LOWER) return c;
		if (firstTail == Tail.UPPER && secondTail == Tail.UPPER) return 1.0-firstLevel-secondLevel+c;
		if (firstTail == Tail.LOWER) return firstLevel-c;
		return secondLevel-c;
	}
	/** Tail-weighted pseudo log likelihood for comparison/fitting objectives. */
	public static NumericalEstimate weightedLogLikelihood(Copula copula, double[][] uniforms,
			double tailPower) {
		if (copula == null || uniforms == null || !(tailPower >= 0.0))
			throw new IllegalArgumentException("copula/data and nonnegative tailPower required");
		double total = 0.0, weights = 0.0;
		for (double[] row : uniforms) {
			if (row == null || row.length != copula.dimension()) throw new IllegalArgumentException("row dimension mismatch");
			double extremeness = 0.0;
			for (double value : row) extremeness = Math.max(extremeness, Math.abs(2.0 * value - 1.0));
			double weight = Math.pow(extremeness, tailPower);
			total += weight * copula.logDensity(row); weights += weight;
		}
		return new NumericalEstimate(total, 0.0, Double.isFinite(total), uniforms.length,
				"tail-weighted-log-likelihood", weights == 0.0 ? "all weights are zero" : "");
	}
	private static NumericalEstimate coefficient(Copula copula, boolean lower) {
		if (copula == null || copula.dimension() != 2) throw new IllegalArgumentException("bivariate copula required");
		double a = lower ? lowerConcentration(copula, 1e-4) : upperConcentration(copula, 1.0-1e-4);
		double b = lower ? lowerConcentration(copula, 5e-5) : upperConcentration(copula, 1.0-5e-5);
		return new NumericalEstimate(Math.max(0.0, Math.min(1.0, b)), Math.abs(a-b), true, 2,
				"finite-level-tail-extrapolation", "coefficient estimated at finite tail levels");
	}
	private static void require(Copula copula, double level) {
		if (copula == null || copula.dimension()!=2 || !(level > 0.0 && level < 1.0))
			throw new IllegalArgumentException("bivariate copula and level in (0,1) required");
	}
}
