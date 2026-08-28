/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Wraps factors without changing whether analytic gradients are available. */
public final class FactorProfiler {
	private FactorProfiler() {}
	public static ProfiledModelFactor profile(String name, ModelFactor factor) {
		if (name == null || factor == null) throw new IllegalArgumentException("name and factor are required");
		return factor instanceof DifferentiableModelFactor ? new DifferentiableProfile(name, (DifferentiableModelFactor) factor) : new ValueProfile(name, factor);
	}
	private abstract static class Base implements ProfiledModelFactor { final String name; long calls, elapsed, nonFinite, allocated;
		Base(String name) { this.name = name; } long usedMemory() { Runtime runtime = Runtime.getRuntime(); return runtime.totalMemory() - runtime.freeMemory(); }
		void record(long start, long memory, double value) { elapsed += System.nanoTime() - start; calls++; allocated += Math.max(0L, usedMemory() - memory); if (!Double.isFinite(value)) nonFinite++; }
		@Override public FactorProfile profile() { return new FactorProfile(name, calls, elapsed, nonFinite, allocated); } }
	private static final class ValueProfile extends Base { private final ModelFactor delegate; ValueProfile(String name, ModelFactor delegate) { super(name); this.delegate = delegate; }
		@Override public synchronized double logDensity(ModelState state) { long memory = usedMemory(), start = System.nanoTime(); double result = delegate.logDensity(state); record(start, memory, result); return result; } }
	private static final class DifferentiableProfile extends Base implements DifferentiableModelFactor { private final DifferentiableModelFactor delegate;
		DifferentiableProfile(String name, DifferentiableModelFactor delegate) { super(name); this.delegate = delegate; }
		@Override public synchronized double logDensityAndAddGradient(ModelState state, double[] gradient) { long memory = usedMemory(), start = System.nanoTime(); double result = delegate.logDensityAndAddGradient(state, gradient); record(start, memory, result); return result; } }
}
