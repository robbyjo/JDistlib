/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/**
 * A copula on the unit hypercube.
 *
 * <p>Implementations are immutable. Density methods are defined on the open
 * unit hypercube; use {@link #diagnose(double[])} before interpreting a density
 * at an exact boundary point.</p>
 */
public interface Copula {
	/** Number of coordinates. */
	int dimension();

	/** Copula distribution function at {@code u}. */
	double cumulative(double[] u);

	/** Natural logarithm of the copula density at an interior point. */
	double logDensity(double[] u);

	/** Copula density at an interior point. */
	default double density(double[] u) {
		return Math.exp(logDensity(u));
	}

	/** Generates one vector of dependent uniform variates. */
	double[] random(RandomEngine random);

	/** Generates one vector using a new deterministic stream. */
	default double[] random(long seed) {
		return random(new MersenneTwister(seed));
	}

	/** Generates {@code count} vectors from one stream. */
	default double[][] random(int count, RandomEngine random) {
		if (count < 0) throw new IllegalArgumentException("sample size must be nonnegative");
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[][] result = new double[count][];
		for (int i = 0; i < count; i++) result[i] = random(random);
		return result;
	}

	/** Generates {@code count} vectors using a new deterministic stream. */
	default double[][] random(int count, long seed) {
		return random(count, new MersenneTwister(seed));
	}

	/** Kendall's tau for a coordinate pair. */
	double kendallsTau(int first, int second);

	/** Matrix of all pairwise Kendall's tau values. */
	default double[][] kendallsTau() {
		int d = dimension();
		double[][] result = new double[d][d];
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < d; j++) result[i][j] = kendallsTau(i, j);
		}
		return result;
	}

	/** Classifies the point for safe CDF and density interpretation. */
	default CopulaDiagnostics diagnose(double[] u) {
		return CopulaDiagnostics.inspect(u, dimension());
	}
}
