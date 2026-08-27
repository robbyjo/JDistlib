/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Immutable named numeric data supplied to a model. */
public final class ModelData {
	private final Map<String, double[]> values;

	ModelData(Map<String, double[]> source) {
		Map<String, double[]> copy = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : source.entrySet())
			copy.put(entry.getKey(), entry.getValue().clone());
		values = Collections.unmodifiableMap(copy);
	}

	public boolean contains(String name) { return values.containsKey(name); }
	public double scalar(String name) {
		double[] result = required(name);
		if (result.length != 1) throw new IllegalArgumentException(name + " is not scalar");
		return result[0];
	}
	public double[] vector(String name) { return required(name).clone(); }
	/** Copies a vector into caller-owned storage without creating a temporary array. */
	public void vectorInto(String name, double[] destination) {
		double[] source = required(name);
		if (destination == null || destination.length != source.length)
			throw new IllegalArgumentException("destination length must equal " + name + " length");
		System.arraycopy(source, 0, destination, 0, source.length);
	}
	public int size() { return values.size(); }
	public Map<String, double[]> asMap() {
		Map<String, double[]> copy = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : values.entrySet())
			copy.put(entry.getKey(), entry.getValue().clone());
		return Collections.unmodifiableMap(copy);
	}
	private double[] required(String name) {
		double[] result = values.get(name);
		if (result == null) throw new IllegalArgumentException("unknown data: " + name);
		return result;
	}
	/* ModelData owns these arrays, so package factors may read them without copying. */
	double[] values(String name) { return required(name); }
}
