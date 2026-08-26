/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/**
 * Joint distribution composed from a copula and continuous univariate
 * marginals.
 *
 * <p>The caller is responsible for supplying continuous marginals. Discrete or
 * mixed marginals require finite-difference mass calculations and are not part
 * of this API.</p>
 */
public final class CopulaDistribution {
	private final Copula copula;
	private final GenericDistribution[] marginals;

	public CopulaDistribution(Copula copula, GenericDistribution... marginals) {
		if (copula == null) throw new IllegalArgumentException("copula must not be null");
		if (marginals == null || marginals.length != copula.dimension()) {
			throw new IllegalArgumentException("one marginal is required per copula coordinate");
		}
		for (GenericDistribution marginal : marginals) {
			if (marginal == null) throw new IllegalArgumentException("marginals must not contain null");
		}
		this.copula = copula;
		this.marginals = marginals.clone();
	}

	public int dimension() { return marginals.length; }
	public Copula getCopula() { return copula; }

	public GenericDistribution getMarginal(int coordinate) {
		if (coordinate < 0 || coordinate >= marginals.length)
			throw new IndexOutOfBoundsException("marginal coordinate out of range");
		return marginals[coordinate];
	}

	/** Joint lower-orthant CDF. */
	public double cumulative(double[] x) {
		return copula.cumulative(toUniforms(x));
	}

	/** Natural logarithm of the joint density. */
	public double logDensity(double[] x) {
		if (x == null || x.length != dimension()) return Double.NaN;
		double[] u = new double[dimension()];
		double result = 0.0;
		for (int i = 0; i < dimension(); i++) {
			u[i] = marginals[i].cumulative(x[i], true, false);
			result += marginals[i].density(x[i], true);
		}
		return result + copula.logDensity(u);
	}

	/** Joint density. */
	public double density(double[] x) { return Math.exp(logDensity(x)); }

	/** Diagnoses the point after transformation through the marginal CDFs. */
	public CopulaDiagnostics diagnose(double[] x) {
		return copula.diagnose(toUniforms(x));
	}

	/** Generates one joint observation. */
	public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[] result = copula.random(random);
		for (int i = 0; i < result.length; i++) {
			result[i] = marginals[i].quantile(result[i], true, false);
		}
		return result;
	}

	/** Generates one observation using a new deterministic stream. */
	public double[] random(long seed) { return random(new MersenneTwister(seed)); }

	/** Generates observations from one stream. */
	public double[][] random(int count, RandomEngine random) {
		if (count < 0) throw new IllegalArgumentException("sample size must be nonnegative");
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[][] result = new double[count][];
		for (int i = 0; i < count; i++) result[i] = random(random);
		return result;
	}

	/** Generates observations using a new deterministic stream. */
	public double[][] random(int count, long seed) {
		return random(count, new MersenneTwister(seed));
	}

	private double[] toUniforms(double[] x) {
		if (x == null || x.length != dimension()) return null;
		double[] result = new double[dimension()];
		for (int i = 0; i < dimension(); i++) {
			result[i] = marginals[i].cumulative(x[i], true, false);
		}
		return result;
	}
}
