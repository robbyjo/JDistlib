/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Fits candidate families and ranks them by AIC or BIC. */
public final class CopulaSelector {
	private CopulaSelector() {}

	public static CopulaSelectionResult select(double[][] data) {
		return select(data, new CopulaFitOptions(), CopulaSelectionCriterion.BIC,
				CopulaFamily.values());
	}

	public static CopulaSelectionResult select(double[][] data,
			CopulaFitOptions options, CopulaSelectionCriterion criterion,
			CopulaFamily... candidates) {
		double[][] uniforms;
		try {
			uniforms = CopulaFitter.pseudoObservations(data);
		} catch (IllegalArgumentException exception) {
			return new CopulaSelectionResult(criterion, new ArrayList<CopulaFitResult>(), null);
		}
		return selectUniforms(uniforms, options, criterion, candidates);
	}

	public static CopulaSelectionResult selectUniforms(double[][] uniforms,
			CopulaFitOptions options, CopulaSelectionCriterion criterion,
			CopulaFamily... candidates) {
		if (options == null || criterion == null || candidates == null
				|| candidates.length == 0)
			throw new IllegalArgumentException("selection options and candidates are required");
		List<CopulaFitResult> results = new ArrayList<>();
		for (CopulaFamily family : Arrays.asList(candidates))
			results.add(CopulaFitter.fitUniforms(uniforms, family, options));
		Comparator<CopulaFitResult> comparator = Comparator.comparingDouble(
				result -> criterion == CopulaSelectionCriterion.AIC
				? result.aic() : result.bic());
		results.sort(comparator);
		CopulaFitResult selected = null;
		for (CopulaFitResult result : results) {
			if (result.isSuccess()) { selected = result; break; }
		}
		return new CopulaSelectionResult(criterion, results, selected);
	}

	/** Selects a family after continuous/discrete marginal transformation. */
	public static CopulaSelectionResult selectMixed(double[][] data,
			CopulaMarginal[] marginals, RandomEngine random,
			CopulaFitOptions options, CopulaSelectionCriterion criterion,
			CopulaFamily... candidates) {
		double[][] uniforms = CopulaFitter.marginalTransforms(data, random, marginals);
		return selectUniforms(uniforms, options, criterion, candidates);
	}
}
