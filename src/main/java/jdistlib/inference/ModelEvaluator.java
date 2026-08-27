/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Non-thread-safe allocation-free evaluator intended for one sampler chain. */
public final class ModelEvaluator implements DifferentiableLogDensity, GradientProvider {
	private final BayesianModel model;
	private final double[] constrained;
	private final double[] constrainedGradient;
	private final ModelState view;

	ModelEvaluator(BayesianModel model) {
		this.model = model;
		constrained = new double[model.constrainedDimension()];
		constrainedGradient = new double[constrained.length];
		view = model.constrainedState(constrained);
	}

	@Override public boolean hasAnalyticGradient() { return model.hasAnalyticGradient(); }

	@Override public double logDensity(double[] state) {
		double result = model.constrainInto(state, constrained);
		for (FactorSpec factor : model.factors()) result += factor.factor().logDensity(view);
		return result;
	}

	@Override public double logDensityAndGradient(double[] state, double[] gradient) {
		Gradients.validate(state, gradient);
		if (!model.hasAnalyticGradient())
			return Gradients.finiteDifference((LogDensity) this)
					.logDensityAndGradient(state, gradient);
		Arrays.fill(constrainedGradient, 0.0);
		double result = model.constrainInto(state, constrained);
		for (FactorSpec factor : model.factors()) result +=
				((DifferentiableModelFactor) factor.factor())
						.logDensityAndAddGradient(view, constrainedGradient);
		for (ParameterSpec parameter : model.parameters().values())
			parameter.constraint().pullback(state, parameter.unconstrainedOffset(),
					constrained, parameter.constrainedOffset(), constrainedGradient, gradient);
		return result;
	}
}
