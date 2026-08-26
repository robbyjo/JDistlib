/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Sequential simplified C-vine and D-vine fitting with pair-family selection. */
public final class VineFitter {
	private VineFitter() {}

	public static VineFitResult fit(double[][] data, VineStructure structure) {
		try {
			return fitUniforms(CopulaFitter.pseudoObservations(data), structure,
					new CopulaFitOptions(), CopulaSelectionCriterion.BIC,
					CopulaFamily.values());
		} catch (IllegalArgumentException exception) {
			return failure(structure, VineFitResult.Status.INVALID_DATA,
					exception.getMessage(), new ArrayList<CopulaSelectionResult>());
		}
	}

	public static VineFitResult fitUniforms(double[][] uniforms,
			VineStructure structure, CopulaFitOptions options,
			CopulaSelectionCriterion criterion, CopulaFamily... candidates) {
		if (structure == null || options == null || criterion == null
				|| candidates == null || candidates.length == 0)
			return failure(structure, VineFitResult.Status.INVALID_DATA,
					"structure, options, criterion, and candidates are required",
					new ArrayList<CopulaSelectionResult>());
		String validation = validateUniforms(uniforms);
		if (validation != null)
			return failure(structure, VineFitResult.Status.INVALID_DATA, validation,
					new ArrayList<CopulaSelectionResult>());
		return structure == VineStructure.C_VINE
				? fitCVine(uniforms, options, criterion, candidates)
				: fitDVine(uniforms, options, criterion, candidates);
	}

	/** Fits a vine after continuous/discrete marginal probability transforms. */
	public static VineFitResult fitMixed(double[][] data, CopulaMarginal[] marginals,
			RandomEngine random, VineStructure structure, CopulaFitOptions options,
			CopulaSelectionCriterion criterion, CopulaFamily... candidates) {
		try {
			return fitUniforms(CopulaFitter.marginalTransforms(data, random, marginals),
					structure, options, criterion, candidates);
		} catch (IllegalArgumentException exception) {
			return failure(structure, VineFitResult.Status.INVALID_DATA,
					exception.getMessage(), new ArrayList<CopulaSelectionResult>());
		}
	}

	private static VineFitResult fitCVine(double[][] uniforms,
			CopulaFitOptions options, CopulaSelectionCriterion criterion,
			CopulaFamily[] candidates) {
		int dimension = uniforms[0].length;
		double[][] conditioned = copy(uniforms);
		PairCopula[][] pairs = new PairCopula[dimension - 1][];
		List<CopulaSelectionResult> selections = new ArrayList<>();
		for (int root = 0; root < dimension - 1; root++) {
			pairs[root] = new PairCopula[dimension - root - 1];
			for (int other = root + 1; other < dimension; other++) {
				double[][] pairData = columns(conditioned, root, other);
				CopulaSelectionResult selection = CopulaSelector.selectUniforms(pairData,
						options, criterion, candidates);
				selections.add(selection);
				if (!selection.isSuccess()) return failure(VineStructure.C_VINE,
						VineFitResult.Status.PAIR_FIT_FAILED,
						"no candidate fit C-vine pair " + root + "," + other, selections);
				PairCopula pair = new PairCopula(selection.getSelected().getCopula());
				pairs[root][other - root - 1] = pair;
				for (int row = 0; row < uniforms.length; row++) {
					conditioned[row][other] = pair.conditionalSecondGivenFirst(
							conditioned[row][root], conditioned[row][other]);
				}
			}
		}
		CVineCopula copula = new CVineCopula(pairs);
		return success(VineStructure.C_VINE, copula, selections, uniforms);
	}

	private static VineFitResult fitDVine(double[][] uniforms,
			CopulaFitOptions options, CopulaSelectionCriterion criterion,
			CopulaFamily[] candidates) {
		int count = uniforms.length;
		int dimension = uniforms[0].length;
		double[][][] direct = new double[count][dimension][dimension];
		double[][][] indirect = new double[count][dimension][dimension];
		for (int row = 0; row < count; row++) {
			for (int i = 0; i < dimension; i++) {
				direct[row][i][i] = uniforms[row][i];
				indirect[row][i][i] = uniforms[row][i];
			}
		}
		PairCopula[][] pairs = new PairCopula[dimension - 1][];
		List<CopulaSelectionResult> selections = new ArrayList<>();
		for (int length = 1; length < dimension; length++) {
			int level = length - 1;
			pairs[level] = new PairCopula[dimension - length];
			for (int first = 0; first + length < dimension; first++) {
				int last = first + length;
				double[][] pairData = new double[count][2];
				for (int row = 0; row < count; row++) {
					pairData[row][0] = indirect[row][first][last - 1];
					pairData[row][1] = direct[row][first + 1][last];
				}
				CopulaSelectionResult selection = CopulaSelector.selectUniforms(pairData,
						options, criterion, candidates);
				selections.add(selection);
				if (!selection.isSuccess()) return failure(VineStructure.D_VINE,
						VineFitResult.Status.PAIR_FIT_FAILED,
						"no candidate fit D-vine interval " + first + "," + last,
						selections);
				PairCopula pair = new PairCopula(selection.getSelected().getCopula());
				pairs[level][first] = pair;
				for (int row = 0; row < count; row++) {
					double left = pairData[row][0];
					double right = pairData[row][1];
					direct[row][first][last] =
							pair.conditionalSecondGivenFirst(left, right);
					indirect[row][first][last] =
							pair.conditionalFirstGivenSecond(left, right);
				}
			}
		}
		DVineCopula copula = new DVineCopula(pairs);
		return success(VineStructure.D_VINE, copula, selections, uniforms);
	}

	private static VineFitResult success(VineStructure structure, VineCopula copula,
			List<CopulaSelectionResult> selections, double[][] uniforms) {
		double logLikelihood = 0.0;
		for (double[] uniform : uniforms) logLikelihood += copula.logDensity(uniform);
		int parameters = 0;
		for (CopulaSelectionResult selection : selections)
			parameters += selection.getSelected().getParameters();
		return new VineFitResult(structure, copula, selections, logLikelihood,
				parameters, VineFitResult.Status.SUCCESS, "vine fit completed");
	}

	private static VineFitResult failure(VineStructure structure,
			VineFitResult.Status status, String message,
			List<CopulaSelectionResult> selections) {
		return new VineFitResult(structure, null, selections, Double.NaN, 0,
				status, message == null ? status.name() : message);
	}

	private static String validateUniforms(double[][] uniforms) {
		if (uniforms == null || uniforms.length < 2 || uniforms[0] == null
				|| uniforms[0].length < 2)
			return "vine data need at least two rows and two columns";
		int dimension = uniforms[0].length;
		for (double[] row : uniforms) {
			if (row == null || row.length != dimension) return "vine data must be rectangular";
			for (double value : row)
				if (!Double.isFinite(value) || !(value > 0.0 && value < 1.0))
					return "vine pseudo-observations must lie in (0, 1)";
		}
		return null;
	}

	private static double[][] copy(double[][] values) {
		double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
		return result;
	}

	private static double[][] columns(double[][] values, int first, int second) {
		double[][] result = new double[values.length][2];
		for (int i = 0; i < values.length; i++) {
			result[i][0] = values[i][first];
			result[i][1] = values[i][second];
		}
		return result;
	}
}
