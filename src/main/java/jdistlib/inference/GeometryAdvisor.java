/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Ranks coordinates whose divergent and non-divergent locations are most separated. */
public final class GeometryAdvisor {
	private GeometryAdvisor() {}
	public static List<GeometryAdvice> analyze(ChainResult... chains) {
		if (chains == null || chains.length == 0) throw new IllegalArgumentException("chains are required"); int dimension = chains[0].dimension();
		double[] divergentSum = new double[dimension], regularSum = new double[dimension], regularSquares = new double[dimension]; int divergentCount = 0, regularCount = 0;
		for (ChainResult chain : chains) for (int draw = 0; draw < chain.size(); draw++) { double[] value = chain.sample(draw);
			if (chain.statisticsAt(draw).divergent()) { divergentCount++; for (int d = 0; d < dimension; d++) divergentSum[d] += value[d]; }
			else { regularCount++; for (int d = 0; d < dimension; d++) { regularSum[d] += value[d]; regularSquares[d] += value[d] * value[d]; } } }
		if (divergentCount == 0 || regularCount < 2) return Collections.emptyList(); List<GeometryAdvice> result = new ArrayList<GeometryAdvice>();
		for (int d = 0; d < dimension; d++) { double regularMean = regularSum[d] / regularCount; double variance = (regularSquares[d] - regularCount * regularMean * regularMean) / (regularCount - 1.0);
			double separation = Math.abs(divergentSum[d] / divergentCount - regularMean) / Math.sqrt(Math.max(1e-12, variance));
			result.add(new GeometryAdvice(d, separation, "Check the scale and transform of unconstrained coordinate " + d + "; consider a non-centered parameterization when it participates in a hierarchy.")); }
		Collections.sort(result, new Comparator<GeometryAdvice>() { @Override public int compare(GeometryAdvice first, GeometryAdvice second) { return -Double.compare(first.standardizedSeparation(), second.standardizedSeparation()); } });
		return Collections.unmodifiableList(result);
	}
}
