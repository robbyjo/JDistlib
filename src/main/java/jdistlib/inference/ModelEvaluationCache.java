/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Cached per-factor values for proposal algorithms that change few coordinates. */
public final class ModelEvaluationCache {
	private final double[] state;
	private final double[] constrained;
	private final double[] proposalConstrained;
	private final double[] factorValues;
	private double value;
	private double jacobian;
	private boolean initialized;

	public ModelEvaluationCache(BayesianModel model) {
		if (model == null) throw new IllegalArgumentException("model is required");
		state = new double[model.dimension()];
		constrained = model.constrain(model.initialState());
		proposalConstrained = new double[constrained.length];
		factorValues = new double[model.factors().size()];
	}

	public double evaluate(BayesianModel model, double[] proposed,
			int... changedCoordinates) {
		if (model == null || proposed == null || proposed.length != state.length)
			throw new IllegalArgumentException("model and matching state are required");
		if (!initialized || changedCoordinates == null || changedCoordinates.length == 0) {
			jacobian = model.constrainInto(proposed, constrained);
			ModelState view = model.constrainedState(constrained);
			value = jacobian;
			for (int i = 0; i < factorValues.length; i++) {
				factorValues[i] = model.factors().get(i).factor().logDensity(view);
				value += factorValues[i];
			}
			System.arraycopy(proposed, 0, state, 0, state.length);
			initialized = true;
			return value;
		}
		double proposedJacobian = model.constrainInto(proposed, proposalConstrained);
		ModelState view = model.constrainedState(proposalConstrained);
		double proposedValue = value - jacobian + proposedJacobian;
		for (int factor = 0; factor < factorValues.length; factor++) {
			FactorSpec spec = model.factors().get(factor);
			boolean affected = false;
			for (int coordinate : changedCoordinates) {
				if (coordinate < 0 || coordinate >= state.length)
					throw new IllegalArgumentException("changed coordinate is outside state");
				affected |= spec.dependsOn(coordinate);
			}
			if (affected) {
				double replacement = spec.factor().logDensity(view);
				proposedValue += replacement - factorValues[factor];
				factorValues[factor] = replacement;
			}
		}
		value = proposedValue;
		jacobian = proposedJacobian;
		System.arraycopy(proposalConstrained, 0, constrained, 0, constrained.length);
		System.arraycopy(proposed, 0, state, 0, state.length);
		return value;
	}

	public double value() {
		if (!initialized) throw new IllegalStateException("cache has not been evaluated");
		return value;
	}
	public double[] state() { return state.clone(); }
}
