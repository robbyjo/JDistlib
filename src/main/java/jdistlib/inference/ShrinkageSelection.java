/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Ranks posterior coefficient draws by practical-significance probability. */
public final class ShrinkageSelection {
	private ShrinkageSelection() {}
	public static Result analyze(String[] variableNames, double[][] coefficientDraws,
			double practicalMagnitude, double minimumProbability) {
		PredictiveMath.requireFiniteMatrix(coefficientDraws, 1, 1);
		int variables = coefficientDraws[0].length;
		if (variableNames == null || variableNames.length != variables || practicalMagnitude < 0.0
				|| !Double.isFinite(practicalMagnitude) || minimumProbability < 0.0 || minimumProbability > 1.0)
			throw new IllegalArgumentException("invalid shrinkage-selection inputs");
		List<Variable> ranking = new ArrayList<Variable>();
		for (int variable = 0; variable < variables; variable++) {
			String name = variableNames[variable];
			if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("variable names required");
			double mean = 0.0, meanAbsolute = 0.0; int beyond = 0;
			for (double[] draw : coefficientDraws) {
				double value = draw[variable]; mean += value; meanAbsolute += Math.abs(value);
				if (Math.abs(value) > practicalMagnitude) beyond++;
			}
			mean /= coefficientDraws.length; meanAbsolute /= coefficientDraws.length;
			double probability = beyond / (double) coefficientDraws.length;
			ranking.add(new Variable(variable, name, mean, meanAbsolute, probability, probability >= minimumProbability));
		}
		Collections.sort(ranking, new Comparator<Variable>() {
			@Override public int compare(Variable first, Variable second) {
				int probability = -Double.compare(first.probability, second.probability);
				return probability != 0 ? probability : -Double.compare(first.meanAbsolute, second.meanAbsolute);
			}
		});
		return new Result(ranking, practicalMagnitude, minimumProbability);
	}
	public static final class Variable {
		private final int index; private final String name; private final double mean, meanAbsolute, probability; private final boolean selected;
		private Variable(int index, String name, double mean, double meanAbsolute, double probability, boolean selected) {
			this.index = index; this.name = name; this.mean = mean; this.meanAbsolute = meanAbsolute; this.probability = probability; this.selected = selected;
		}
		public int index() { return index; }
		public String name() { return name; }
		public double posteriorMean() { return mean; }
		public double posteriorMeanAbsoluteMagnitude() { return meanAbsolute; }
		public double probabilityBeyondPracticalMagnitude() { return probability; }
		public boolean selected() { return selected; }
	}
	public static final class Result {
		private final List<Variable> ranking; private final double practicalMagnitude, minimumProbability;
		private Result(List<Variable> ranking, double practicalMagnitude, double minimumProbability) {
			this.ranking = Collections.unmodifiableList(new ArrayList<Variable>(ranking)); this.practicalMagnitude = practicalMagnitude; this.minimumProbability = minimumProbability;
		}
		public List<Variable> ranking() { return ranking; }
		public List<Variable> selected() {
			List<Variable> result = new ArrayList<Variable>(); for (Variable variable : ranking) if (variable.selected()) result.add(variable);
			return Collections.unmodifiableList(result);
		}
		public double practicalMagnitude() { return practicalMagnitude; }
		public double minimumProbability() { return minimumProbability; }
	}
}
