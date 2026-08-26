/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Simplified C-vine copula assembled from bivariate conditional copulas. */
public final class CVineCopula implements VineCopula {
	private static final long CDF_SEED = 0x4356696e65436466L;
	private static final int DEFAULT_CDF_SAMPLES = 16_384;
	private final PairCopula[][] pairs;

	/**
	 * Creates a C-vine. Row {@code root} contains pair copulas
	 * {@code (root, root+1)}, ..., {@code (root, d-1)} conditioned on all earlier
	 * roots, so row lengths must be {@code d-1, d-2, ..., 1}.
	 */
	public CVineCopula(PairCopula[]... pairs) {
		if (pairs == null || pairs.length == 0)
			throw new IllegalArgumentException("a C-vine needs at least one tree");
		int dimension = pairs.length + 1;
		this.pairs = new PairCopula[pairs.length][];
		for (int root = 0; root < pairs.length; root++) {
			if (pairs[root] == null || pairs[root].length != dimension - root - 1)
				throw new IllegalArgumentException("C-vine tree rows have invalid lengths");
			this.pairs[root] = pairs[root].clone();
			for (PairCopula pair : this.pairs[root])
				if (pair == null) throw new IllegalArgumentException("C-vine pairs must not be null");
		}
	}

	@Override public int dimension() { return pairs.length + 1; }

	public PairCopula getPairCopula(int root, int other) {
		if (root < 0 || other <= root || other >= dimension())
			throw new IndexOutOfBoundsException("invalid C-vine pair index");
		return pairs[root][other - root - 1];
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension())) return Double.NaN;
		double[] conditioned = u.clone();
		double result = 0.0;
		for (int root = 0; root < dimension() - 1; root++) {
			double rootValue = conditioned[root];
			for (int other = root + 1; other < dimension(); other++) {
				PairCopula pair = pairs[root][other - root - 1];
				result += pair.logDensity(rootValue, conditioned[other]);
				conditioned[other] = pair.conditionalSecondGivenFirst(
						rootValue, conditioned[other]);
			}
		}
		return result;
	}

	@Override public double cumulative(double[] u) {
		if (!CopulaUtil.validPoint(u, dimension())) return Double.NaN;
		if (CopulaUtil.hasZero(u)) return 0.0;
		int nonUpper = 0;
		double margin = 1.0;
		for (double value : u) {
			if (value < 1.0) { nonUpper++; margin = value; }
		}
		if (nonUpper == 0) return 1.0;
		if (nonUpper == 1) return margin;
		return cumulativeResult(u, DEFAULT_CDF_SAMPLES,
				new MersenneTwister(CDF_SEED)).probability;
	}

	/** Estimates the lower-orthant CDF with caller-owned randomization. */
	public VineProbabilityResult cumulativeResult(double[] u, int samples,
			RandomEngine random) {
		if (!CopulaUtil.validPoint(u, dimension()) || samples < 1 || random == null)
			return new VineProbabilityResult(Double.NaN, Double.NaN, 0);
		int hits = 0;
		for (int sample = 0; sample < samples; sample++) {
			double[] draw = random(random);
			boolean inside = true;
			for (int i = 0; i < dimension(); i++) {
				if (draw[i] > u[i]) { inside = false; break; }
			}
			if (inside) hits++;
		}
		double estimate = (double) hits / samples;
		double standardError = Math.sqrt(estimate * (1.0 - estimate) / samples);
		return new VineProbabilityResult(estimate, standardError, samples);
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[] innovations = new double[dimension()];
		double[] result = new double[dimension()];
		for (int i = 0; i < dimension(); i++) innovations[i] = CopulaUtil.uniformOpen(random);
		result[0] = innovations[0];
		for (int other = 1; other < dimension(); other++) {
			double value = innovations[other];
			for (int root = other - 1; root >= 0; root--) {
				value = pairs[root][other - root - 1]
						.inverseSecondGivenFirst(innovations[root], value);
			}
			result[other] = value;
		}
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension());
		if (first == second) return 1.0;
		if (first == 0 || second == 0) {
			int other = Math.max(first, second);
			return pairs[0][other - 1].getCopula().kendallsTau(0, 1);
		}
		return empiricalTau(first, second, 3000, new MersenneTwister(CDF_SEED));
	}

	private double empiricalTau(int first, int second, int count,
			RandomEngine random) {
		double[][] sample = random(count, random);
		long concordant = 0;
		long discordant = 0;
		for (int i = 0; i < count; i++) {
			for (int j = i + 1; j < count; j++) {
				double product = (sample[i][first] - sample[j][first])
						* (sample[i][second] - sample[j][second]);
				if (product > 0.0) concordant++;
				else if (product < 0.0) discordant++;
			}
		}
		return (double) (concordant - discordant) / (concordant + discordant);
	}
}
