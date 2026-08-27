/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.Constraints;
import jdistlib.inference.ParameterConstraint;

public class MatrixConstraintTest {
	@Test public void covarianceAndCorrelationTransformsRoundTrip() {
		assertRoundTrip(Constraints.covarianceMatrix(3),
				new double[] {0.2, -0.1, 0.3, 0.4, -0.2, -0.15});
		assertRoundTrip(Constraints.correlationMatrix(4),
				new double[] {0.1, -0.2, 0.3, 0.25, -0.35, 0.15});
		assertRoundTrip(Constraints.choleskyFactorCorrelation(4),
				new double[] {0.1, -0.2, 0.3, 0.25, -0.35, 0.15});
		assertRoundTrip(Constraints.choleskyFactorCovariance(4, 3),
				new double[] {0.2, -0.1, 0.3, 0.4, -0.2, -0.15, 0.7, -0.4, 0.2});
	}

	@Test public void exactPullbacksMatchFiniteDifferences() {
		assertPullback(Constraints.sumToZero(5), new double[] {0.3, -0.7, 0.4, 0.2});
		assertPullback(Constraints.unitVector(3), new double[] {0.3, -0.7, 0.4});
		assertPullback(Constraints.covarianceMatrix(3),
				new double[] {0.2, -0.1, 0.3, 0.4, -0.2, -0.15});
		assertPullback(Constraints.correlationMatrix(4),
				new double[] {0.1, -0.2, 0.3, 0.25, -0.35, 0.15});
		assertPullback(Constraints.choleskyFactorCorrelation(4),
				new double[] {0.1, -0.2, 0.3, 0.25, -0.35, 0.15});
		assertPullback(Constraints.choleskyFactorCovariance(4, 3),
				new double[] {0.2, -0.1, 0.3, 0.4, -0.2, -0.15, 0.7, -0.4, 0.2});
	}

	@Test public void repeatedStructuredTransformsTileArrays() {
		ParameterConstraint repeated = Constraints.repeated(Constraints.correlationMatrix(3), 2);
		double[] unconstrained = {0.1, -0.2, 0.3, -0.15, 0.25, -0.35};
		assertEquals(6, repeated.unconstrainedDimension());
		assertEquals(18, repeated.constrainedDimension());
		assertRoundTrip(repeated, unconstrained);
		assertPullback(repeated, unconstrained);
	}

	private static void assertRoundTrip(ParameterConstraint constraint, double[] unconstrained) {
		double[] constrained = new double[constraint.constrainedDimension()];
		double jacobian = constraint.constrain(unconstrained, 0, constrained, 0);
		assertTrue(Double.isFinite(jacobian));
		double[] recovered = new double[constraint.unconstrainedDimension()];
		constraint.unconstrain(constrained, 0, recovered, 0);
		for (int i = 0; i < recovered.length; i++) assertEquals(unconstrained[i], recovered[i], 1e-9);
	}

	private static void assertPullback(ParameterConstraint constraint, double[] unconstrained) {
		double[] constrained = new double[constraint.constrainedDimension()];
		constraint.constrain(unconstrained, 0, constrained, 0);
		double[] constrainedGradient = new double[constrained.length];
		for (int i = 0; i < constrainedGradient.length; i++)
			constrainedGradient[i] = 0.13 + 0.07 * i;
		double[] analytic = new double[unconstrained.length];
		constraint.pullback(unconstrained, 0, constrained, 0, constrainedGradient, analytic);
		double epsilon = 1e-6;
		for (int coordinate = 0; coordinate < unconstrained.length; coordinate++) {
			double[] plus = unconstrained.clone(), minus = unconstrained.clone();
			plus[coordinate] += epsilon; minus[coordinate] -= epsilon;
			double numeric = (objective(constraint, plus, constrainedGradient)
					- objective(constraint, minus, constrainedGradient)) / (2.0 * epsilon);
			assertEquals("coordinate " + coordinate + " of " + constraint.description(),
					numeric, analytic[coordinate], 2e-6);
		}
	}

	private static double objective(ParameterConstraint constraint, double[] unconstrained,
			double[] constrainedGradient) {
		double[] constrained = new double[constraint.constrainedDimension()];
		double result = constraint.constrain(unconstrained, 0, constrained, 0);
		for (int i = 0; i < constrained.length; i++) result += constrained[i] * constrainedGradient[i];
		return result;
	}
}
