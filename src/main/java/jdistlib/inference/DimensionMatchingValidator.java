/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Executable round-trip, dimension, and reciprocal-Jacobian validation for RJ maps. */
public final class DimensionMatchingValidator {
	private DimensionMatchingValidator() {}
	public static boolean validate(DimensionMatchingTransformation transformation,
			ReversibleJumpState state, double[] auxiliary, double tolerance) {
		if (transformation == null || state == null || auxiliary == null || tolerance < 0.0 || !Double.isFinite(tolerance))
			throw new IllegalArgumentException("transformation, input, and tolerance required");
		DimensionMatchingResult forward = transformation.forward(state, auxiliary);
		if (state.dimension() + auxiliary.length != forward.state().dimension() + forward.auxiliary().length) return false;
		DimensionMatchingResult inverse = transformation.inverse(forward.state(), forward.auxiliary());
		if (inverse.state().modelId() != state.modelId() || !close(inverse.state().parameters(), state.parameters(), tolerance)
				|| !close(inverse.auxiliary(), auxiliary, tolerance)) return false;
		double first = transformation.logAbsJacobian(state, auxiliary, true);
		double second = transformation.logAbsJacobian(forward.state(), forward.auxiliary(), false);
		return Double.isFinite(first) && Double.isFinite(second) && Math.abs(first + second) <= tolerance;
	}
	private static boolean close(double[] first, double[] second, double tolerance) {
		if (first.length != second.length) return false;
		for (int i = 0; i < first.length; i++) if (Math.abs(first[i] - second[i]) > tolerance) return false;
		return true;
	}
}
