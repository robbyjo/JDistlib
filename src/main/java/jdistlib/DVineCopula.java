/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Simplified D-vine copula assembled from bivariate conditional copulas. */
public final class DVineCopula implements VineCopula {
	private static final long CDF_SEED = 0x4456696e65436466L;
	private static final int DEFAULT_CDF_SAMPLES = 16_384;
	private final PairCopula[][] pairs;

	/**
	 * Creates a D-vine. Tree {@code level} contains pairs
	 * {@code (i, i+level+1 | i+1,...,i+level)} and therefore has
	 * {@code d-level-1} entries.
	 */
	public DVineCopula(PairCopula[]... pairs) {
		if (pairs == null || pairs.length == 0)
			throw new IllegalArgumentException("a D-vine needs at least one tree");
		int dimension = pairs.length + 1;
		this.pairs = new PairCopula[pairs.length][];
		for (int level = 0; level < pairs.length; level++) {
			if (pairs[level] == null || pairs[level].length != dimension - level - 1)
				throw new IllegalArgumentException("D-vine tree rows have invalid lengths");
			this.pairs[level] = pairs[level].clone();
			for (PairCopula pair : this.pairs[level])
				if (pair == null) throw new IllegalArgumentException("D-vine pairs must not be null");
		}
	}

	@Override public int dimension() { return pairs.length + 1; }

	public PairCopula getPairCopula(int level, int first) {
		if (level < 0 || level >= pairs.length || first < 0
				|| first >= pairs[level].length)
			throw new IndexOutOfBoundsException("invalid D-vine pair index");
		return pairs[level][first];
	}

	@Override public double logDensity(double[] u) {
		if (!CopulaUtil.interiorPoint(u, dimension())) return Double.NaN;
		double[][] direct = new double[dimension()][dimension()];
		double[][] indirect = new double[dimension()][dimension()];
		for (int i = 0; i < dimension(); i++) {
			direct[i][i] = u[i];
			indirect[i][i] = u[i];
		}
		double result = 0.0;
		for (int length = 1; length < dimension(); length++) {
			int level = length - 1;
			for (int first = 0; first + length < dimension(); first++) {
				int last = first + length;
				double left = indirect[first][last - 1];
				double right = direct[first + 1][last];
				PairCopula pair = pairs[level][first];
				result += pair.logDensity(left, right);
				direct[first][last] = pair.conditionalSecondGivenFirst(left, right);
				indirect[first][last] = pair.conditionalFirstGivenSecond(left, right);
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

	@Override public VineProbabilityResult cumulativeResult(double[] u, int samples,
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
		return new VineProbabilityResult(estimate,
				Math.sqrt(estimate * (1.0 - estimate) / samples), samples);
	}

	@Override public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[][] direct = new double[dimension()][dimension()];
		double[][] indirect = new double[dimension()][dimension()];
		double[] result = new double[dimension()];
		result[0] = CopulaUtil.uniformOpen(random);
		direct[0][0] = result[0];
		indirect[0][0] = result[0];
		for (int last = 1; last < dimension(); last++) {
			double value = CopulaUtil.uniformOpen(random);
			for (int first = 0; first < last; first++) {
				int level = last - first - 1;
				value = pairs[level][first].inverseSecondGivenFirst(
						indirect[first][last - 1], value);
			}
			result[last] = value;
			direct[last][last] = value;
			indirect[last][last] = value;
			for (int first = last - 1; first >= 0; first--) {
				int level = last - first - 1;
				double left = indirect[first][last - 1];
				double right = direct[first + 1][last];
				PairCopula pair = pairs[level][first];
				direct[first][last] = pair.conditionalSecondGivenFirst(left, right);
				indirect[first][last] = pair.conditionalFirstGivenSecond(left, right);
			}
		}
		return result;
	}

	@Override public double kendallsTau(int first, int second) {
		CopulaUtil.requirePair(first, second, dimension());
		if (first == second) return 1.0;
		if (Math.abs(first - second) == 1) {
			int lower = Math.min(first, second);
			return pairs[0][lower].getCopula().kendallsTau(0, 1);
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
