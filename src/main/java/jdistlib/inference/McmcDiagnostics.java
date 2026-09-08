/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdistlib.Normal;

/** Rank-normalized R-hat, bulk/tail ESS, MCSE, and sampler diagnostics. */
public final class McmcDiagnostics {
	private McmcDiagnostics() {}

	/** Nested R-hat for chains grouped by common-initialization superchain IDs. */
	public static double nestedRHat(double[][] chains, int[] superchainIds) {
		validateNested(chains, superchainIds);
		Map<Integer, List<Integer>> groups = groups(superchainIds);
		double[] chainMeans = new double[chains.length], chainVariances = new double[chains.length];
		for (int chain = 0; chain < chains.length; chain++) {
			chainMeans[chain] = mean(chains[chain]);
			chainVariances[chain] = sampleVariance(chains[chain], chainMeans[chain]);
		}
		double[] superMeans = new double[groups.size()]; double withinSuper = 0.0; int groupIndex = 0;
		for (List<Integer> group : groups.values()) {
			double[] means = new double[group.size()]; double withinChain = 0.0;
			for (int i = 0; i < group.size(); i++) { int chain = group.get(i); means[i] = chainMeans[chain]; withinChain += chainVariances[chain]; }
			withinChain /= group.size(); double groupMean = mean(means); superMeans[groupIndex++] = groupMean;
			double betweenChain = group.size() == 1 ? 0.0 : sampleVariance(means, groupMean);
			withinSuper += withinChain + betweenChain;
		}
		withinSuper /= groups.size(); double betweenSuper = sampleVariance(superMeans, mean(superMeans));
		if (!(withinSuper > 0.0) || !Double.isFinite(withinSuper)) return Double.NaN;
		return Math.sqrt(Math.max(1.0, 1.0 + betweenSuper / withinSuper));
	}

	/** Rank-normalized and folded nested R-hat, matching the robust ordinary diagnostic. */
	public static double nestedRankNormalizedRHat(double[][] chains, int[] superchainIds) {
		validateNested(chains, superchainIds); double[] pooled = flatten(chains);
		double median = quantile(pooled, 0.5);
		return Math.max(nestedRHat(rankNormalize(chains), superchainIds),
				nestedRHat(rankNormalize(fold(chains, median)), superchainIds));
	}

	private static void validateNested(double[][] chains, int[] ids) {
		if (chains == null || chains.length < 2 || ids == null || ids.length != chains.length)
			throw new IllegalArgumentException("chains and one superchain ID per chain are required");
		int draws = -1; for (double[] chain : chains) {
			if (chain == null || chain.length == 0 || (draws >= 0 && chain.length != draws))
				throw new IllegalArgumentException("nested R-hat requires rectangular, nonempty chains");
			draws = chain.length; for (double value : chain) if (!Double.isFinite(value)) return;
		}
		Map<Integer, List<Integer>> groups = groups(ids); if (groups.size() < 2)
			throw new IllegalArgumentException("at least two superchains are required");
		int size = -1; for (List<Integer> group : groups.values()) { if (size < 0) size = group.size();
			else if (group.size() != size) throw new IllegalArgumentException("superchains must contain equal chain counts"); }
	}
	private static Map<Integer, List<Integer>> groups(int[] ids) {
		Map<Integer, List<Integer>> result = new LinkedHashMap<Integer, List<Integer>>();
		for (int i = 0; i < ids.length; i++) { List<Integer> group = result.get(ids[i]);
			if (group == null) { group = new ArrayList<Integer>(); result.put(ids[i], group); } group.add(i); }
		return result;
	}
	private static double sampleVariance(double[] values, double center) {
		if (values.length < 2) return 0.0; double result = 0.0;
		for (double value : values) { double difference = value - center; result += difference * difference; }
		return result / (values.length - 1.0);
	}

	public static McmcDiagnosticReport analyze(ChainResult... chains) {
		if (chains == null || chains.length == 0)
			throw new IllegalArgumentException("at least one chain is required");
		int dimension = chains[0].dimension();
		String[] names = new String[dimension];
		for (int i = 0; i < dimension; i++) names[i] = "state[" + i + "]";
		return analyze(names, chains);
	}

	public static McmcDiagnosticReport analyze(String[] names, ChainResult... chains) {
		if (chains == null || chains.length == 0 || names == null)
			throw new IllegalArgumentException("names and chains are required");
		int dimension = chains[0].dimension();
		if (names.length != dimension) throw new IllegalArgumentException("one name is required per coordinate");
		int draws = Integer.MAX_VALUE;
		for (ChainResult chain : chains) {
			if (chain == null || chain.dimension() != dimension)
				throw new IllegalArgumentException("chain dimensions must match");
			draws = Math.min(draws, chain.size());
		}
		if (draws < 4) throw new IllegalArgumentException("at least four retained draws per chain are required");
		List<ParameterDiagnostics> parameters = new ArrayList<ParameterDiagnostics>();
		List<String> warnings = new ArrayList<String>();
		for (int coordinate = 0; coordinate < dimension; coordinate++) {
			double[][] values = coordinate(chains, coordinate, draws);
			double[] pooled = flatten(values);
			double mean = mean(pooled);
			double sd = standardDeviation(pooled, mean);
			double median = quantile(pooled, 0.5);
			double lower = quantile(pooled, 0.025);
			double upper = quantile(pooled, 0.975);
			double[][] ranks = rankNormalize(split(values));
			double rHat = chains.length < 2 ? Double.NaN
					: Math.max(rHat(ranks), rHat(rankNormalize(split(fold(values, median)))));
			double bulkEss = effectiveSampleSize(ranks);
			double lowThreshold = quantile(pooled, 0.05);
			double highThreshold = quantile(pooled, 0.95);
			double tailEss = Math.min(effectiveSampleSize(split(indicator(values, lowThreshold, true))),
					effectiveSampleSize(split(indicator(values, highThreshold, true))));
			double mcse = sd / Math.sqrt(effectiveSampleSize(split(values)));
			boolean reliable = (Double.isNaN(rHat) || rHat < 1.01)
					&& bulkEss >= 100.0 && tailEss >= 100.0;
			if (!reliable) warnings.add(names[coordinate]
					+ " has insufficient convergence or effective sample size");
			parameters.add(new ParameterDiagnostics(names[coordinate], mean, sd,
					median, lower, upper, rHat, bulkEss, tailEss, mcse, reliable));
		}
		SamplerDiagnostics sampler = samplerDiagnostics(chains, draws);
		if (sampler.divergences() > 0) warnings.add(sampler.divergences() + " divergent transitions detected");
		if (sampler.treeDepthSaturations() > 0) warnings.add(sampler.treeDepthSaturations() + " maximum-tree-depth transitions detected");
		if (Double.isFinite(sampler.energyBayesianFractionMissingInformation())
				&& sampler.energyBayesianFractionMissingInformation() < 0.3)
			warnings.add("low energy Bayesian fraction of missing information");
		return new McmcDiagnosticReport(parameters, sampler, warnings,
				chains.length, draws);
	}

	static double autocorrelation(double[] values, int lag) {
		if (lag < 0 || lag >= values.length) return Double.NaN;
		double mean = mean(values);
		double variance = 0.0;
		double covariance = 0.0;
		for (int i = 0; i < values.length; i++) {
			double centered = values[i] - mean;
			variance += centered * centered;
			if (i + lag < values.length) covariance += centered * (values[i + lag] - mean);
		}
		return variance == 0.0 ? (lag == 0 ? 1.0 : 0.0) : covariance / variance;
	}

	private static SamplerDiagnostics samplerDiagnostics(ChainResult[] chains, int draws) {
		double acceptance = 0.0;
		int acceptanceCount = 0;
		int divergences = 0;
		int saturated = 0;
		int maxDepth = 0;
		int failures = 0;
		double minimumEbfmi = Double.POSITIVE_INFINITY;
		boolean hasEbfmi = false;
		for (ChainResult chain : chains) {
			if (chain.status() == ChainResult.Status.NUMERICAL_FAILURE
					|| chain.status() == ChainResult.Status.INVALID_INITIAL_STATE) failures++;
			int start = Math.max(0, chain.size() - draws);
			List<Double> energies = new ArrayList<Double>();
			for (int i = start; i < chain.size(); i++) {
				IterationStats stat = chain.statisticsAt(i);
				if (Double.isFinite(stat.acceptanceProbability())) {
					acceptance += stat.acceptanceProbability(); acceptanceCount++;
				}
				if (stat.divergent()) divergences++;
				if (stat.treeDepthSaturated()) saturated++;
				maxDepth = Math.max(maxDepth, stat.treeDepth());
				if (Double.isFinite(stat.energy())) energies.add(stat.energy());
			}
			double chainEbfmi = energyBfmi(energies);
			if (Double.isFinite(chainEbfmi)) {
				hasEbfmi = true;
				minimumEbfmi = Math.min(minimumEbfmi, chainEbfmi);
			}
		}
		double ebfmi = hasEbfmi ? minimumEbfmi : Double.NaN;
		return new SamplerDiagnostics(acceptanceCount == 0 ? Double.NaN
				: acceptance / acceptanceCount, divergences, saturated, maxDepth,
				ebfmi, failures);
	}

	private static double energyBfmi(List<Double> energies) {
		if (energies.size() < 3) return Double.NaN;
		double mean = 0.0;
		for (double energy : energies) mean += energy;
		mean /= energies.size();
		double variance = 0.0;
		double differences = 0.0;
		for (int i = 0; i < energies.size(); i++) {
			double centered = energies.get(i) - mean;
			variance += centered * centered;
			if (i > 0) {
				double difference = energies.get(i) - energies.get(i - 1);
				differences += difference * difference;
			}
		}
		return variance == 0.0 ? Double.NaN : differences / variance;
	}

	private static double effectiveSampleSize(double[][] chains) {
		int m = chains.length, n = chains[0].length;
		if (n < 3 || invalidOrConstant(chains)) return Double.NaN;
		double[] means = new double[m], covariance = new double[n];
		for (int chain = 0; chain < m; chain++) {
			means[chain] = mean(chains[chain]);
			double[] current = autocovariances(chains[chain], means[chain]);
			for (int lag = 0; lag < n; lag++) covariance[lag] += current[lag] / m;
		}
		double meanVariance = covariance[0] * n / (n - 1.0);
		double variancePlus = covariance[0] + (m > 1 ? sampleVariance(means, mean(means)) : 0);
		if (!(variancePlus > 0)) return Double.NaN;
		// Geyer initial-positive and initial-monotone sequence, including the
		// final positive even lag and the antithetic ESS cap used by posterior.
		double[] rho = new double[n];
		double even = 1, odd = 1 - (meanVariance - covariance[1]) / variancePlus;
		rho[0] = even; rho[1] = odd;
		int t = 0;
		while (t < n - 5 && even + odd > 0) {
			t += 2;
			even = 1 - (meanVariance - covariance[t]) / variancePlus;
			odd = 1 - (meanVariance - covariance[t + 1]) / variancePlus;
			if (even + odd >= 0) { rho[t] = even; rho[t + 1] = odd; }
		}
		int maxT = t;
		if (even > 0) rho[maxT] = even;
		for (t = 2; t <= maxT - 2; t += 2) {
			double previous = rho[t - 2] + rho[t - 1];
			if (rho[t] + rho[t + 1] > previous) rho[t] = rho[t + 1] = previous / 2;
		}
		double sum = maxT == 0 ? rho[0] : 0;
		for (t = 0; t < maxT; t++) sum += rho[t];
		double total = (double) m * n, tau = -1 + 2 * sum + rho[maxT];
		return total / Math.max(1 / Math.log10(total), tau);
	}

	private static double[] autocovariances(double[] values, double center) {
		int n = values.length, padded = 1;
		while (padded < 2L * n) padded *= 2;
		double[] spectrum = new double[2 * padded];
		for (int i = 0; i < n; i++) spectrum[i] = values[i] - center;
		org.jtransforms.fft.DoubleFFT_1D fft = new org.jtransforms.fft.DoubleFFT_1D(padded);
		fft.realForwardFull(spectrum);
		for (int i = 0; i < padded; i++) {
			double real = spectrum[2 * i], imaginary = spectrum[2 * i + 1];
			spectrum[2 * i] = real * real + imaginary * imaginary;
			spectrum[2 * i + 1] = 0;
		}
		fft.complexInverse(spectrum, true);
		double[] result = new double[n];
		for (int i = 0; i < n; i++) result[i] = spectrum[2 * i] / n;
		return result;
	}

	private static boolean invalidOrConstant(double[][] values) {
		double first = values[0][0]; boolean constant = true;
		for (double[] chain : values) for (double value : chain) {
			if (!Double.isFinite(value)) return true;
			if (value != first) constant = false;
		}
		return constant;
	}

	private static double rHat(double[][] values) {
		if (invalidOrConstant(values)) return Double.NaN;
		int m = values.length;
		int n = values[0].length;
		double[] means = new double[m];
		double within = 0.0;
		for (int chain = 0; chain < m; chain++) {
			means[chain] = mean(values[chain]);
			double variance = 0.0;
			for (double value : values[chain]) {
				double centered = value - means[chain]; variance += centered * centered;
			}
			within += variance / (n - 1.0);
		}
		within /= m;
		double meanOfMeans = mean(means);
		double between = 0.0;
		for (double value : means) between += (value - meanOfMeans) * (value - meanOfMeans);
		between *= n / (m - 1.0);
		if (within == 0.0) return between == 0.0 ? 1.0 : Double.POSITIVE_INFINITY;
		double variance = (n - 1.0) / n * within + between / n;
		return Math.sqrt(variance / within);
	}

	private static double[][] split(double[][] values) {
		int half = values[0].length / 2;
		double[][] result = new double[values.length * 2][half];
		for (int chain = 0; chain < values.length; chain++) {
			System.arraycopy(values[chain], 0, result[2 * chain], 0, half);
			System.arraycopy(values[chain], values[chain].length - half,
					result[2 * chain + 1], 0, half);
		}
		return result;
	}

	private static double[][] rankNormalize(double[][] values) {
		int m = values.length;
		int n = values[0].length;
		final double[] pooled = flatten(values);
		for (double value : pooled) if (!Double.isFinite(value)) {
			double[][] missing = new double[m][n]; for (double[] chain : missing) Arrays.fill(chain, Double.NaN); return missing;
		}
		Integer[] order = new Integer[pooled.length];
		for (int i = 0; i < order.length; i++) order[i] = i;
		Arrays.sort(order, new Comparator<Integer>() {
			@Override public int compare(Integer first, Integer second) {
				return Double.compare(pooled[first], pooled[second]);
			}
		});
		double[] ranks = new double[pooled.length];
		for (int start = 0; start < order.length;) {
			int end = start + 1;
			while (end < order.length && pooled[order[start]] == pooled[order[end]]) end++;
			double rank = 0.5 * (start + end - 1) + 1.0;
			for (int i = start; i < end; i++) ranks[order[i]] = rank;
			start = end;
		}
		double[][] result = new double[m][n];
		for (int i = 0; i < ranks.length; i++) {
			double probability = (ranks[i] - 0.375) / (ranks.length + 0.25);
			result[i / n][i % n] = Normal.quantile(probability, 0.0, 1.0, true, false);
		}
		return result;
	}

	private static double[][] fold(double[][] values, double median) {
		double[][] result = new double[values.length][values[0].length];
		for (int i = 0; i < values.length; i++)
			for (int j = 0; j < values[i].length; j++)
				result[i][j] = Math.abs(values[i][j] - median);
		return result;
	}
	private static double[][] indicator(double[][] values, double threshold, boolean lower) {
		double[][] result = new double[values.length][values[0].length];
		for (int i = 0; i < values.length; i++)
			for (int j = 0; j < values[i].length; j++)
				result[i][j] = lower ? (values[i][j] <= threshold ? 1.0 : 0.0)
						: (values[i][j] >= threshold ? 1.0 : 0.0);
		return result;
	}
	private static double[][] coordinate(ChainResult[] chains, int coordinate, int draws) {
		double[][] result = new double[chains.length][draws];
		for (int chain = 0; chain < chains.length; chain++) {
			int start = chains[chain].size() - draws;
			for (int draw = 0; draw < draws; draw++)
				result[chain][draw] = chains[chain].valueAt(start + draw, coordinate);
		}
		return result;
	}
	private static double[] flatten(double[][] values) {
		double[] result = new double[values.length * values[0].length];
		int offset = 0;
		for (double[] value : values) {
			System.arraycopy(value, 0, result, offset, value.length); offset += value.length;
		}
		return result;
	}
	private static double mean(double[] values) {
		double result = 0.0;
		for (double value : values) result += value;
		return result / values.length;
	}
	private static double standardDeviation(double[] values, double mean) {
		double result = 0.0;
		for (double value : values) result += (value - mean) * (value - mean);
		return Math.sqrt(result / Math.max(1.0, values.length - 1.0));
	}
	private static double quantile(double[] values, double probability) {
		double[] sorted = values.clone(); Arrays.sort(sorted);
		double index = probability * (sorted.length - 1.0);
		int lower = (int) Math.floor(index);
		int upper = Math.min(sorted.length - 1, lower + 1);
		return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
	}
}
