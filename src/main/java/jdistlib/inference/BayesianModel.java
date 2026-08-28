/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Compiled named model evaluated on an unconstrained state space. */
public final class BayesianModel implements DifferentiableLogDensity, GradientProvider,
		PointwiseLogLikelihood {
	private final Map<String, ParameterSpec> parameters;
	private final List<FactorSpec> factors;
	private final ModelData data;
	private final double[] initialState;
	private final int constrainedDimension;
	private final boolean analyticGradient;
	private final ModelGraph graph;
	private final List<PointwiseLikelihoodSpec> pointwise;
	private final ObservationMetadata observationMetadata;

	BayesianModel(Map<String, ParameterSpec> parameters, List<FactorSpec> factors,
			ModelData data, double[] initialState, int constrainedDimension) {
		this(parameters, factors, data, initialState, constrainedDimension,
				Collections.<PointwiseLikelihoodSpec>emptyList());
	}

	BayesianModel(Map<String, ParameterSpec> parameters, List<FactorSpec> factors,
			ModelData data, double[] initialState, int constrainedDimension,
			List<PointwiseLikelihoodSpec> pointwise) {
		this.parameters = Collections.unmodifiableMap(
				new LinkedHashMap<String, ParameterSpec>(parameters));
		this.factors = Collections.unmodifiableList(new ArrayList<FactorSpec>(factors));
		this.data = data;
		this.initialState = initialState.clone();
		this.constrainedDimension = constrainedDimension;
		boolean analytic = true;
		for (FactorSpec factor : factors) analytic &= factor.isDifferentiable();
		analyticGradient = analytic;
		this.pointwise = Collections.unmodifiableList(
				new ArrayList<PointwiseLikelihoodSpec>(pointwise));
		ObservationMetadata[] pieces = new ObservationMetadata[this.pointwise.size()];
		for (int i = 0; i < pieces.length; i++) pieces[i] = this.pointwise.get(i).metadata;
		observationMetadata = ObservationMetadata.concatenate(pieces);
		graph = createGraph();
	}

	public int dimension() { return initialState.length; }
	public double[] initialState() { return initialState.clone(); }
	public Map<String, ParameterSpec> parameters() { return parameters; }
	public List<FactorSpec> factors() { return factors; }
	public ModelData data() { return data; }
	public ModelGraph graph() { return graph; }
	public ModelEvaluator evaluator() { return new ModelEvaluator(this); }
	@Override public boolean hasAnalyticGradient() { return analyticGradient; }
	@Override public ObservationMetadata observationMetadata() { return observationMetadata; }

	@Override public double[] pointwiseLogLikelihood(double[] state) {
		if (state == null || state.length != dimension())
			throw new IllegalArgumentException("state dimension does not match model");
		double[] constrained = constrain(state);
		ModelState view = new ModelState(constrained, parameters, data);
		double[] result = new double[observationMetadata.size()]; int offset = 0;
		for (PointwiseLikelihoodSpec spec : pointwise) {
			double[] values = spec.evaluator.evaluate(view);
			if (values == null || values.length != spec.metadata.size())
				throw new IllegalStateException("pointwise evaluator width does not match its metadata");
			System.arraycopy(values, 0, result, offset, values.length); offset += values.length;
		}
		return result;
	}

	/** Returns a copy of this model with an additional pointwise likelihood evaluator. */
	public BayesianModel withPointwiseLikelihood(ObservationMetadata metadata,
			PointwiseLogLikelihoodEvaluator evaluator) {
		if (metadata == null || metadata.size() == 0 || evaluator == null)
			throw new IllegalArgumentException("nonempty metadata and evaluator are required");
		List<PointwiseLikelihoodSpec> combined = new ArrayList<PointwiseLikelihoodSpec>(pointwise);
		combined.add(new PointwiseLikelihoodSpec(metadata, evaluator));
		return new BayesianModel(parameters, factors, data, initialState,
				constrainedDimension, combined);
	}

	@Override public double logDensity(double[] state) {
		if (state == null || state.length != dimension()) return Double.NaN;
		double[] constrained = new double[constrainedDimension];
		double result = constrain(state, constrained);
		ModelState view = new ModelState(constrained, parameters, data);
		for (FactorSpec factor : factors) result += factor.factor().logDensity(view);
		return result;
	}

	@Override public double logDensityAndGradient(double[] state, double[] gradient) {
		Gradients.validate(state, gradient);
		if (state.length != dimension())
			throw new IllegalArgumentException("state dimension does not match model");
		if (!analyticGradient) {
			return Gradients.finiteDifference((LogDensity) this)
					.logDensityAndGradient(state, gradient);
		}
		double[] constrained = new double[constrainedDimension];
		double result = constrain(state, constrained);
		double[] constrainedGradient = new double[constrainedDimension];
		ModelState view = new ModelState(constrained, parameters, data);
		for (FactorSpec factor : factors) {
			result += ((DifferentiableModelFactor) factor.factor())
					.logDensityAndAddGradient(view, constrainedGradient);
		}
		for (ParameterSpec parameter : parameters.values()) {
			parameter.constraint().pullback(state, parameter.unconstrainedOffset(),
					constrained, parameter.constrainedOffset(), constrainedGradient, gradient);
		}
		return result;
	}

	public double[] constrain(double[] state) {
		if (state == null || state.length != dimension())
			throw new IllegalArgumentException("state dimension does not match model");
		double[] result = new double[constrainedDimension];
		constrain(state, result);
		return result;
	}

	/** Converts a complete map of named constrained parameter values to sampler space. */
	public double[] unconstrain(Map<String, double[]> values) {
		if (values == null) throw new IllegalArgumentException("values are required");
		double[] result = new double[dimension()];
		for (ParameterSpec parameter : parameters.values()) {
			double[] value = values.get(parameter.name());
			if (value == null || value.length != parameter.constrainedDimension())
				throw new IllegalArgumentException("missing or invalid constrained value: "
						+ parameter.name());
			parameter.constraint().unconstrain(value, 0, result,
					parameter.unconstrainedOffset());
		}
		return result;
	}

	/** Returns a named constrained view backed by a fresh transformed state. */
	public ModelState state(double[] unconstrainedState) {
		return new ModelState(constrain(unconstrainedState), parameters, data);
	}

	double constrainInto(double[] state, double[] constrained) {
		if (state == null || state.length != dimension()
				|| constrained == null || constrained.length != constrainedDimension)
			throw new IllegalArgumentException("model transform dimensions do not match");
		return constrain(state, constrained);
	}

	ModelState constrainedState(double[] constrained) {
		return new ModelState(constrained, parameters, data);
	}

	int constrainedDimension() { return constrainedDimension; }

	private double constrain(double[] state, double[] constrained) {
		double jacobian = 0.0;
		for (ParameterSpec parameter : parameters.values()) {
			jacobian += parameter.constraint().constrain(state,
					parameter.unconstrainedOffset(), constrained,
					parameter.constrainedOffset());
		}
		return jacobian;
	}

	private ModelGraph createGraph() {
		List<ModelGraph.Node> nodes = new ArrayList<ModelGraph.Node>();
		List<ModelGraph.Edge> edges = new ArrayList<ModelGraph.Edge>();
		for (ParameterSpec parameter : parameters.values())
			nodes.add(new ModelGraph.Node("parameter:" + parameter.name(),
					parameter.name() + " : " + parameter.constraint().description(),
					ModelGraph.NodeKind.PARAMETER));
		for (String datum : data.asMap().keySet())
			nodes.add(new ModelGraph.Node("data:" + datum, datum, ModelGraph.NodeKind.DATA));
		for (FactorSpec factor : factors) {
			String factorId = "factor:" + factor.name();
			nodes.add(new ModelGraph.Node(factorId, factor.name(), ModelGraph.NodeKind.FACTOR));
			for (String dependency : factor.dependencies()) {
				String source = parameters.containsKey(dependency)
						? "parameter:" + dependency : "data:" + dependency;
				edges.add(new ModelGraph.Edge(source, factorId));
			}
		}
		return new ModelGraph(nodes, edges);
	}
}
