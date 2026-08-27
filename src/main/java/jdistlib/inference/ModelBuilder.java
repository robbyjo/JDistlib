/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdistlib.generic.GenericDistribution;

/** Fluent builder for named constrained parameters and model factors. */
public final class ModelBuilder {
	private static final class PendingParameter {
		final String name;
		final ParameterConstraint constraint;
		final double[] initial;
		PendingParameter(String name, ParameterConstraint constraint, double[] initial) {
			this.name = name; this.constraint = constraint; this.initial = initial;
		}
	}
	private static final class PendingFactor {
		final String name;
		final String[] dependencies;
		final ModelFactor factor;
		PendingFactor(String name, String[] dependencies, ModelFactor factor) {
			this.name = name; this.dependencies = dependencies; this.factor = factor;
		}
	}

	private final Map<String, PendingParameter> parameters =
			new LinkedHashMap<String, PendingParameter>();
	private final Map<String, double[]> data = new LinkedHashMap<String, double[]>();
	private final List<PendingFactor> factors = new ArrayList<PendingFactor>();

	public ModelBuilder data(String name, double... values) {
		validateName(name);
		if (parameters.containsKey(name) || data.containsKey(name) || values == null)
			throw new IllegalArgumentException("duplicate name or null data: " + name);
		data.put(name, values.clone());
		return this;
	}

	public ModelBuilder parameter(String name, ParameterConstraint constraint,
			double... initialConstrainedValue) {
		validateName(name);
		if (constraint == null || initialConstrainedValue == null
				|| initialConstrainedValue.length != constraint.constrainedDimension()
				|| parameters.containsKey(name) || data.containsKey(name)) {
			throw new IllegalArgumentException("invalid or duplicate parameter: " + name);
		}
		parameters.put(name, new PendingParameter(name, constraint,
				initialConstrainedValue.clone()));
		return this;
	}

	public ModelBuilder factor(String name, String[] dependencies,
			ModelFactor factor) {
		if (name == null || name.trim().isEmpty() || factor == null || dependencies == null)
			throw new IllegalArgumentException("factor and dependencies are required");
		for (PendingFactor existing : factors)
			if (existing.name.equals(name)) throw new IllegalArgumentException("duplicate factor: " + name);
		factors.add(new PendingFactor(name, dependencies.clone(), factor));
		return this;
	}

	/** Adds an independent fixed scalar prior for a scalar parameter. */
	public ModelBuilder prior(final String parameter, final GenericDistribution prior) {
		if (prior == null) throw new IllegalArgumentException("prior is required");
		return factor(parameter + "~" + prior.getClass().getSimpleName(),
				new String[] {parameter}, state -> prior.density(state.scalar(parameter), true));
	}

	public BayesianModel build() {
		if (parameters.isEmpty()) throw new IllegalStateException("at least one parameter is required");
		Map<String, ParameterSpec> compiled = new LinkedHashMap<String, ParameterSpec>();
		int unconstrainedDimension = 0;
		int constrainedDimension = 0;
		for (PendingParameter parameter : parameters.values()) {
			compiled.put(parameter.name, new ParameterSpec(parameter.name,
					parameter.constraint, unconstrainedDimension, constrainedDimension));
			unconstrainedDimension += parameter.constraint.unconstrainedDimension();
			constrainedDimension += parameter.constraint.constrainedDimension();
		}
		List<FactorSpec> compiledFactors = new ArrayList<FactorSpec>();
		for (PendingFactor factor : factors) {
			boolean[] mask = new boolean[unconstrainedDimension];
			for (String dependency : factor.dependencies) {
				ParameterSpec parameter = compiled.get(dependency);
				if (parameter == null && !data.containsKey(dependency))
					throw new IllegalStateException("factor " + factor.name
							+ " has unknown dependency " + dependency);
				if (parameter != null) Arrays.fill(mask, parameter.unconstrainedOffset(),
						parameter.unconstrainedOffset() + parameter.unconstrainedDimension(), true);
			}
			compiledFactors.add(new FactorSpec(factor.name, factor.dependencies,
					factor.factor, mask));
		}
		double[] initial = new double[unconstrainedDimension];
		for (PendingParameter pending : parameters.values()) {
			ParameterSpec spec = compiled.get(pending.name);
			spec.constraint().unconstrain(pending.initial, 0, initial,
					spec.unconstrainedOffset());
		}
		return new BayesianModel(compiled, compiledFactors, new ModelData(data),
				initial, constrainedDimension);
	}

	private static void validateName(String name) {
		if (name == null || !name.matches("[A-Za-z_][A-Za-z0-9_]*"))
			throw new IllegalArgumentException("invalid model name: " + name);
	}
}
