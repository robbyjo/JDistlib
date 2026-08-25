/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.MultivariateProbabilityResult;
import jdistlib.MultivariateProbabilityStatus;
import jdistlib.MultivariateNormal;
import jdistlib.Normal;

/** Minimal packaged-JAR smoke test intentionally limited to Java 8 syntax. */
public final class Java8JarSmoke {
	private Java8JarSmoke() {}

	public static void main(String[] arguments) {
		double probability = Normal.cumulative(0.0, 0.0, 1.0, true, false);
		if (probability != 0.5) throw new AssertionError("normal CDF smoke failed");
		MultivariateProbabilityResult result = MultivariateNormal.cumulative(
				new double[] {0.0}, new double[] {0.0}, new double[][] {{1.0}});
		if (result.getStatus() != MultivariateProbabilityStatus.SUCCESS ||
				Math.abs(result.probability - 0.5) > 1e-15)
			throw new AssertionError("multivariate CDF smoke failed");
	}
}
