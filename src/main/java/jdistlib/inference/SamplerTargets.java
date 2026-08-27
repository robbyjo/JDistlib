/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Selects chain-local evaluators so model buffers are reused safely. */
final class SamplerTargets {
	private SamplerTargets() {}
	static LogDensity local(LogDensity target) {
		return target instanceof BayesianModel
				? ((BayesianModel) target).evaluator() : target;
	}
}
