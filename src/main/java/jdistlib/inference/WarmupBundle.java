/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Fingerprinted metric/step-size warmup reuse, treated as an initial guess by default. */
public final class WarmupBundle {
	private final double stepSize; private final double[][] inverseMass; private final String modelFingerprint, geometryFingerprint;
	private WarmupBundle(double stepSize, double[][] inverseMass, String modelFingerprint, String geometryFingerprint) {
		this.stepSize = stepSize; this.inverseMass = copy(inverseMass); this.modelFingerprint = modelFingerprint; this.geometryFingerprint = geometryFingerprint; }
	public static WarmupBundle from(WarmupResult warmup, String modelFingerprint, String geometryFingerprint) {
		if (warmup == null || warmup.inverseMassMatrix() == null || modelFingerprint == null || geometryFingerprint == null) throw new IllegalArgumentException("complete warmup and fingerprints are required");
		return new WarmupBundle(warmup.finalStepSize(), warmup.inverseMassMatrix(), modelFingerprint, geometryFingerprint); }
	public SamplingOptions apply(SamplingOptions options, String expectedModel, String expectedGeometry, boolean fixAdaptation) {
		if (!modelFingerprint.equals(expectedModel) || !geometryFingerprint.equals(expectedGeometry)) throw new IllegalArgumentException("warmup fingerprint mismatch");
		return options.toBuilder().stepSize(stepSize).metric(options.metricConfiguration().withInitialInverseMassMatrix(inverseMass))
				.adaptStepSize(!fixAdaptation).adaptMassMatrix(!fixAdaptation).build(); }
	public double stepSize() { return stepSize; } public double[][] inverseMassMatrix() { return copy(inverseMass); }
	public String modelFingerprint() { return modelFingerprint; } public String geometryFingerprint() { return geometryFingerprint; }
	private static double[][] copy(double[][] values) { if (values == null) return null; double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone(); return result; }
}
