/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.Dirichlet;
import jdistlib.DirichletMultinomial;
import jdistlib.Multinomial;
import jdistlib.MultivariateHypergeometric;
import jdistlib.MultivariateLaplace;
import jdistlib.MultivariatePowerExponential;
import jdistlib.MultivariateProbabilityResult;
import jdistlib.Wiener;
import jdistlib.rng.MersenneTwister;

/** Executable tour of exact count and error-reporting continuous probabilities. */
public final class AdvancedProbabilityExamples {
	private AdvancedProbabilityExamples() {}

	public static void main(String[] arguments) {
		int[] lowerCounts = {1, 0, 2};
		int[] upperCounts = {4, 3, 6};
		System.out.println("multinomial rectangle=" + Multinomial.probability(
				lowerCounts, upperCounts, 7, new double[] {0.2, 0.3, 0.5}));
		System.out.println("Dirichlet-multinomial rectangle=" +
				DirichletMultinomial.probability(lowerCounts, upperCounts, 7,
						new double[] {0.7, 1.4, 2.1}));
		System.out.println("multivariate hypergeometric rectangle=" +
				MultivariateHypergeometric.probability(lowerCounts, upperCounts,
						new int[] {5, 7, 9}, 7));

		report("Dirichlet simplex rectangle", Dirichlet.probability(
				new double[] {0.1, 0.1, 0.1}, new double[] {0.7, 0.6, 0.8},
				new double[] {2.0, 3.0, 4.0}));
		double[] location = {0.0, 0.0};
		double[][] scatter = {{1.0, 0.5}, {0.5, 1.0}};
		report("multivariate Laplace lower orthant",
				MultivariateLaplace.cumulative(new double[] {0.0, 0.0}, location,
						scatter));
		report("power-exponential rectangle",
				MultivariatePowerExponential.probability(new double[] {-1.0, -0.5},
						new double[] {0.8, 1.2}, location, scatter, 0.65));

		double logDensity = Wiener.density(0.75, 1.2, 0.18, 0.35, 0.6, true);
		double conditionalDraw = Wiener.random(1.2, 0.18, 0.35, 0.6,
				new MersenneTwister(20260828L));
		System.out.println("Wiener log density=" + logDensity +
				", conditional upper-response draw=" + conditionalDraw);
		System.out.println("count bounds=" + Arrays.toString(lowerCounts) + " .. " +
				Arrays.toString(upperCounts));
	}

	private static void report(String label, MultivariateProbabilityResult result) {
		System.out.println(label + "=" + result.probability + " +/- " +
				result.absoluteError + " (" + result.getStatus() + ")");
	}
}
