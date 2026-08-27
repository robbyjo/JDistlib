/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable factor metadata and evaluator. */
public final class FactorSpec {
	private final String name;
	private final String[] dependencies;
	private final ModelFactor factor;
	private final boolean[] unconstrainedDependencies;

	FactorSpec(String name, String[] dependencies, ModelFactor factor,
			boolean[] unconstrainedDependencies) {
		this.name = name;
		this.dependencies = dependencies.clone();
		this.factor = factor;
		this.unconstrainedDependencies = unconstrainedDependencies;
	}
	public String name() { return name; }
	public String[] dependencies() { return dependencies.clone(); }
	public ModelFactor factor() { return factor; }
	public boolean isDifferentiable() { return factor instanceof DifferentiableModelFactor; }
	boolean dependsOn(int coordinate) { return unconstrainedDependencies[coordinate]; }
}
