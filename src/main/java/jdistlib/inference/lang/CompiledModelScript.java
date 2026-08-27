/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ModelState;
import jdistlib.rng.RandomEngine;

/** Compiled model plus deterministic or random generated-quantity program. */
public final class CompiledModelScript {
	interface Generator {
		Map<String, double[]> generate(ModelState state, RandomEngine random);
	}
	private final BayesianModel model;
	private final Generator generator;
	private final String languageVersion;
	CompiledModelScript(BayesianModel model, Generator generator, String languageVersion) {
		this.model = model; this.generator = generator; this.languageVersion = languageVersion;
	}
	public BayesianModel model() { return model; }
	public String languageVersion() { return languageVersion; }
	public Map<String, double[]> generate(double[] unconstrainedState, RandomEngine random) {
		if (unconstrainedState == null || random == null)
			throw new IllegalArgumentException("state and random stream are required");
		Map<String, double[]> values = generator.generate(model.state(unconstrainedState), random);
		Map<String, double[]> copy = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : values.entrySet())
			copy.put(entry.getKey(), entry.getValue().clone());
		return Collections.unmodifiableMap(copy);
	}
}
