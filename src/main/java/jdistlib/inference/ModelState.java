/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Map;

/** Read-only named view of one constrained model state and its observed data. */
public final class ModelState {
	private final double[] values;
	private final Map<String, ParameterSpec> parameters;
	private final ModelData data;

	ModelState(double[] values, Map<String, ParameterSpec> parameters, ModelData data) {
		this.values = values;
		this.parameters = parameters;
		this.data = data;
	}

	public double scalar(String name) {
		ParameterSpec parameter = required(name);
		if (parameter.constrainedDimension() != 1)
			throw new IllegalArgumentException(name + " is not scalar");
		return values[parameter.constrainedOffset()];
	}

	public double value(String name, int index) {
		ParameterSpec parameter = required(name);
		if (index < 0 || index >= parameter.constrainedDimension())
			throw new IndexOutOfBoundsException("parameter index out of range");
		return values[parameter.constrainedOffset() + index];
	}

	public double[] vector(String name) {
		ParameterSpec parameter = required(name);
		double[] result = new double[parameter.constrainedDimension()];
		System.arraycopy(values, parameter.constrainedOffset(), result, 0, result.length);
		return result;
	}

	public ModelData data() { return data; }
	public int constrainedDimension() { return values.length; }
	public boolean hasParameter(String name) { return parameters.containsKey(name); }
	public int parameterDimension(String name) { return required(name).constrainedDimension(); }
	public int constrainedOffset(String name) { return required(name).constrainedOffset(); }
	public void addGradient(String name, int index, double amount, double[] gradient) {
		ParameterSpec parameter = required(name);
		if (gradient == null || gradient.length != values.length || index < 0
				|| index >= parameter.constrainedDimension())
			throw new IllegalArgumentException("invalid model-state gradient target");
		gradient[parameter.constrainedOffset() + index] += amount;
	}

	private ParameterSpec required(String name) {
		ParameterSpec parameter = parameters.get(name);
		if (parameter == null) throw new IllegalArgumentException("unknown parameter: " + name);
		return parameter;
	}
}
