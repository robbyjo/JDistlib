/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Dispatches to a distinct within-model kernel for each declared model identifier. */
public final class ModelSpecificRjKernel implements ReversibleJumpWithinModelKernel {
	private final String name; private final Map<Long, ReversibleJumpWithinModelKernel> kernels;
	public ModelSpecificRjKernel(String name, Map<Long, ReversibleJumpWithinModelKernel> kernels) {
		if (name == null || name.trim().isEmpty() || kernels == null || kernels.isEmpty()) throw new IllegalArgumentException("name and model kernels required");
		Map<Long, ReversibleJumpWithinModelKernel> copy = new LinkedHashMap<Long, ReversibleJumpWithinModelKernel>();
		for (Map.Entry<Long, ReversibleJumpWithinModelKernel> entry : kernels.entrySet())
			if (entry.getKey() == null || entry.getKey().longValue() < 0L || entry.getValue() == null) throw new IllegalArgumentException("valid model kernels required");
			else copy.put(entry.getKey(), entry.getValue());
		this.name = name; this.kernels = Collections.unmodifiableMap(copy);
	}
	@Override public String name() { return name; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) {
		ReversibleJumpWithinModelKernel kernel = state == null ? null : kernels.get(Long.valueOf(state.modelId()));
		return kernel != null && kernel.applicable(state, target);
	}
	@Override public ReversibleJumpWithinModelTransition update(ReversibleJumpState state, double currentLogJoint,
			ReversibleJumpTarget target, RandomEngine random, boolean warmup) {
		ReversibleJumpWithinModelKernel kernel = kernels.get(Long.valueOf(state.modelId()));
		if (kernel == null) throw new IllegalArgumentException("no kernel for model " + state.modelId());
		return kernel.update(state, currentLogJoint, target, random, warmup);
	}
	@Override public Map<String, double[]> adaptationState() {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		for (Map.Entry<Long, ReversibleJumpWithinModelKernel> entry : kernels.entrySet()) for (Map.Entry<String, double[]> value : entry.getValue().adaptationState().entrySet())
			result.put(name + "/" + entry.getKey() + "/" + value.getKey(), value.getValue());
		return result;
	}
	@Override public void restoreAdaptation(Map<String, double[]> state) {
		for (Map.Entry<Long, ReversibleJumpWithinModelKernel> entry : kernels.entrySet()) {
			String prefix = name + "/" + entry.getKey() + "/"; Map<String, double[]> nested = new LinkedHashMap<String, double[]>();
			for (Map.Entry<String, double[]> value : state.entrySet()) if (value.getKey().startsWith(prefix)) nested.put(value.getKey().substring(prefix.length()), value.getValue());
			entry.getValue().restoreAdaptation(nested);
		}
	}
	@Override public void freezeAdaptation() { for (ReversibleJumpWithinModelKernel kernel : kernels.values()) kernel.freezeAdaptation(); }
	@Override public void resetAdaptation() { for (ReversibleJumpWithinModelKernel kernel : kernels.values()) kernel.resetAdaptation(); }
}
