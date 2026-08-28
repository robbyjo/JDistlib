/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Multi-chain diagnostics for ragged reversible-jump output. */
public final class ReversibleJumpDiagnostics {
	private static final class Moments { long count; double mean, products; void add(double value) { count++; double delta = value - mean; mean += delta / count; products += delta * (value - mean); } }
	private ReversibleJumpDiagnostics() {}
	public static ReversibleJumpDiagnosticReport analyze(ReversibleJumpTarget target, ReversibleJumpResult... chains) {
		if (target == null || chains == null || chains.length == 0) throw new IllegalArgumentException("target and RJ chains required");
		Map<Long, Long> visits = new LinkedHashMap<Long, Long>(); Map<String, Moments> parameters = new LinkedHashMap<String, Moments>();
		Map<String, long[]> moves = new LinkedHashMap<String, long[]>(); List<String> warnings = new ArrayList<String>(); long total = 0L;
		for (int chainIndex = 0; chainIndex < chains.length; chainIndex++) {
			ReversibleJumpResult chain = chains[chainIndex]; if (chain == null || chain.size() == 0) throw new IllegalArgumentException("nonempty RJ chains required");
			if (chain.status() != ReversibleJumpResult.Status.SUCCESS) warnings.add("chain " + chainIndex + " status is " + chain.status());
			for (int draw = 0; draw < chain.size(); draw++) {
				ReversibleJumpState state = chain.draw(draw); ReversibleJumpModelSpace space = target.modelSpace(state.modelId());
				Long key = Long.valueOf(state.modelId()); visits.put(key, Long.valueOf(visits.containsKey(key) ? visits.get(key).longValue() + 1L : 1L)); total++;
				for (int parameter = 0; parameter < state.dimension(); parameter++) {
					String name = space.parameterName(parameter); Moments moments = parameters.get(name); if (moments == null) { moments = new Moments(); parameters.put(name, moments); }
					moments.add(state.parameter(parameter));
				}
			}
			for (int move = 0; move < chain.moveCount(); move++) {
				String name = chain.moveName(move); long[] counts = moves.get(name); if (counts == null) { counts = new long[3]; moves.put(name, counts); }
				counts[0] += chain.moveAttempts(move); counts[1] += chain.moveAccepts(move); counts[2] += chain.invalidProposals(move);
			}
		}
		List<Map.Entry<Long, Long>> ordered = new ArrayList<Map.Entry<Long, Long>>(visits.entrySet());
		Collections.sort(ordered, new Comparator<Map.Entry<Long, Long>>() { @Override public int compare(Map.Entry<Long, Long> first, Map.Entry<Long, Long> second) { return -Long.compare(first.getValue(), second.getValue()); } });
		long[] modelIds = new long[ordered.size()], modelVisits = new long[ordered.size()];
		double[] probabilities = new double[ordered.size()], modelEss = new double[ordered.size()], modelRHat = new double[ordered.size()], modelMcse = new double[ordered.size()];
		Map<Long, Integer> modelIndex = new LinkedHashMap<Long, Integer>();
		for (int i = 0; i < ordered.size(); i++) {
			modelIds[i] = ordered.get(i).getKey().longValue(); modelVisits[i] = ordered.get(i).getValue().longValue(); probabilities[i] = modelVisits[i] / (double) total;
			double[][] indicator = modelIndicators(chains, modelIds[i]); modelEss[i] = indicatorEss(indicator, probabilities[i]); modelRHat[i] = indicatorRHat(indicator);
			modelMcse[i] = Math.sqrt(probabilities[i] * (1.0 - probabilities[i]) / modelEss[i]); modelIndex.put(Long.valueOf(modelIds[i]), Integer.valueOf(i));
			if (modelVisits[i] < 20L) warnings.add("model " + modelIds[i] + " has fewer than 20 retained visits");
			int chainsVisited = 0; for (ReversibleJumpResult chain : chains) if (containsModel(chain, modelIds[i])) chainsVisited++;
			if (chainsVisited < chains.length) warnings.add("model " + modelIds[i] + " was visited by only " + chainsVisited + " of " + chains.length + " chains");
			if (modelEss[i] < 100.0) warnings.add("model " + modelIds[i] + " has indicator ESS below 100");
			if (!Double.isNaN(modelRHat[i]) && modelRHat[i] > 1.01) warnings.add("model " + modelIds[i] + " has indicator R-hat above 1.01");
		}
		long[][] transitions = new long[modelIds.length][modelIds.length]; long modelChanges = 0L;
		for (ReversibleJumpResult chain : chains) for (int draw = 1; draw < chain.size(); draw++) {
			int from = modelIndex.get(Long.valueOf(chain.draw(draw - 1).modelId())).intValue(), to = modelIndex.get(Long.valueOf(chain.draw(draw).modelId())).intValue();
			transitions[from][to]++; if (from != to) modelChanges++;
		}
		if (modelChanges == 0L) warnings.add("no retained model changes were observed");
		int roundTrips = modelIds.length < 2 ? 0 : roundTrips(modelIds[0], modelIds[1], chains);
		String[] candidateNames = new String[0]; double[] inclusion = new double[0], ess = new double[0], rhat = new double[0], mcse = new double[0];
		if (target instanceof SubsetSelectionTarget) {
			SubsetSelectionTarget subset = (SubsetSelectionTarget) target; candidateNames = subset.candidateNames(); int candidates = candidateNames.length;
			inclusion = new double[candidates]; ess = new double[candidates]; rhat = new double[candidates]; mcse = new double[candidates];
			for (int candidate = 0; candidate < candidates; candidate++) {
				double[][] indicators = indicators(chains, candidate); double sum = 0.0; for (double[] chain : indicators) for (double value : chain) sum += value;
				inclusion[candidate] = sum / total; ess[candidate] = indicatorEss(indicators, inclusion[candidate]);
				rhat[candidate] = indicatorRHat(indicators); mcse[candidate] = Math.sqrt(inclusion[candidate] * (1.0 - inclusion[candidate]) / ess[candidate]);
				if (ess[candidate] < 100.0) warnings.add("candidate " + candidateNames[candidate] + " has inclusion ESS below 100");
				if (!Double.isNaN(rhat[candidate]) && rhat[candidate] > 1.01) warnings.add("candidate " + candidateNames[candidate] + " has inclusion R-hat above 1.01");
			}
		}
		String[] moveNames = moves.keySet().toArray(new String[moves.size()]); long[] attempts = new long[moves.size()], accepts = new long[moves.size()], invalid = new long[moves.size()];
		for (int i = 0; i < moveNames.length; i++) { long[] counts = moves.get(moveNames[i]); attempts[i] = counts[0]; accepts[i] = counts[1]; invalid[i] = counts[2];
			if (attempts[i] == 0L) warnings.add("move " + moveNames[i] + " was never attempted");
			else { double rate = accepts[i] / (double) attempts[i]; if (rate < 0.01) warnings.add("move " + moveNames[i] + " acceptance is below 1%"); }
		}
		ReversibleJumpParameterSummary[] summaries = new ReversibleJumpParameterSummary[parameters.size()]; int summary = 0;
		for (Map.Entry<String, Moments> entry : parameters.entrySet()) { Moments value = entry.getValue(); summaries[summary++] = new ReversibleJumpParameterSummary(entry.getKey(), value.count, value.mean, value.count < 2 ? 0.0 : Math.sqrt(value.products / (value.count - 1.0))); }
		return new ReversibleJumpDiagnosticReport(modelIds, modelVisits, probabilities, modelEss, modelRHat, modelMcse,
				transitions, candidateNames, inclusion,
				ess, rhat, mcse, moveNames, attempts, accepts, invalid, summaries, modelChanges, roundTrips, warnings);
	}
	private static double[][] indicators(ReversibleJumpResult[] chains, int candidate) {
		double[][] result = new double[chains.length][]; long mask = 1L << candidate;
		for (int chain = 0; chain < chains.length; chain++) { result[chain] = new double[chains[chain].size()]; for (int draw = 0; draw < result[chain].length; draw++) result[chain][draw] = (chains[chain].draw(draw).modelId() & mask) == 0L ? 0.0 : 1.0; }
		return result;
	}
	private static double[][] modelIndicators(ReversibleJumpResult[] chains, long modelId) {
		double[][] result = new double[chains.length][];
		for (int chain = 0; chain < chains.length; chain++) { result[chain] = new double[chains[chain].size()];
			for (int draw = 0; draw < result[chain].length; draw++) result[chain][draw] = chains[chain].draw(draw).modelId() == modelId ? 1.0 : 0.0; }
		return result;
	}
	private static boolean containsModel(ReversibleJumpResult chain, long modelId) {
		for (int draw = 0; draw < chain.size(); draw++) if (chain.draw(draw).modelId() == modelId) return true;
		return false;
	}
	private static double indicatorEss(double[][] chains, double mean) {
		double denominator = 0.0; int total = 0, maximumLag = Integer.MAX_VALUE;
		for (double[] chain : chains) { total += chain.length; maximumLag = Math.min(maximumLag, chain.length - 1); for (double value : chain) denominator += (value - mean) * (value - mean); }
		if (denominator == 0.0) return total; maximumLag = Math.min(maximumLag, 1000); double sum = 0.0;
		for (int lag = 1; lag <= maximumLag; lag += 2) {
			double pair = autocorrelation(chains, mean, denominator, lag); if (lag + 1 <= maximumLag) pair += autocorrelation(chains, mean, denominator, lag + 1);
			if (pair <= 0.0) break; sum += pair;
		}
		return Math.max(1.0, Math.min(total, total / Math.max(1.0, 1.0 + 2.0 * sum)));
	}
	private static double autocorrelation(double[][] chains, double mean, double denominator, int lag) {
		double numerator = 0.0; for (double[] chain : chains) for (int i = lag; i < chain.length; i++) numerator += (chain[i] - mean) * (chain[i - lag] - mean); return numerator / denominator;
	}
	private static double indicatorRHat(double[][] chains) {
		int common = Integer.MAX_VALUE; for (double[] chain : chains) common = Math.min(common, chain.length); int half = common / 2;
		if (chains.length < 2 || half < 2) return Double.NaN; int segments = chains.length * 2; double[] means = new double[segments], variances = new double[segments]; int segment = 0;
		for (double[] chain : chains) for (int split = 0; split < 2; split++) { int start = split == 0 ? 0 : common - half; double mean = 0.0; for (int i = 0; i < half; i++) mean += chain[start + i]; mean /= half; means[segment] = mean;
			double variance = 0.0; for (int i = 0; i < half; i++) { double difference = chain[start + i] - mean; variance += difference * difference; } variances[segment++] = variance / (half - 1.0); }
		double mean = 0.0, within = 0.0; for (int i = 0; i < segments; i++) { mean += means[i]; within += variances[i]; } mean /= segments; within /= segments;
		double between = 0.0; for (double value : means) { double difference = value - mean; between += difference * difference; } between *= half / (segments - 1.0);
		if (within == 0.0) return between == 0.0 ? 1.0 : Double.POSITIVE_INFINITY;
		return Math.sqrt(((half - 1.0) / half * within + between / half) / within);
	}
	private static int roundTrips(long first, long second, ReversibleJumpResult[] chains) {
		int trips = 0; for (ReversibleJumpResult chain : chains) { long anchor = -1L; boolean crossed = false;
			for (ReversibleJumpState draw : chain.draws()) { long model = draw.modelId(); if (model != first && model != second) continue;
				if (anchor < 0L) anchor = model; else if (model != anchor) crossed = true; else if (crossed) { trips++; crossed = false; } }
		} return trips;
	}
}
