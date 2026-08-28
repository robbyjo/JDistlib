/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Immutable provenance record for reproducing one inference run. */
public final class RunManifest {
	private final String libraryVersion, sampler, modelHash, optionsHash;
	private final long seed, startedEpochMillis, elapsedNanoseconds;
	private RunManifest(String libraryVersion, String sampler, String modelHash,
			String optionsHash, long seed, long startedEpochMillis, long elapsedNanoseconds) {
		this.libraryVersion = libraryVersion; this.sampler = sampler;
		this.modelHash = modelHash; this.optionsHash = optionsHash; this.seed = seed;
		this.startedEpochMillis = startedEpochMillis; this.elapsedNanoseconds = elapsedNanoseconds;
	}
	public static RunManifest create(Sampler sampler, String modelIdentity,
			SamplingOptions options, long seed, long startedEpochMillis,
			long elapsedNanoseconds) {
		if (sampler == null || modelIdentity == null || options == null)
			throw new IllegalArgumentException("manifest inputs are required");
		Package library = RunManifest.class.getPackage();
		String version = library == null ? null : library.getImplementationVersion();
		if (version == null) version = "development";
		String optionText = options.warmupIterations() + ":" + options.sampleIterations()
				+ ":" + options.thinning() + ":" + options.stepSize() + ":"
				+ options.targetAcceptance() + ":" + options.leapfrogSteps() + ":"
				+ options.maximumTreeDepth() + ":" + options.metricConfiguration().type()
				+ ":" + options.maximumEnergyError() + ":" + options.integrationTime()
				+ ":" + options.stepSizeJitter() + ":" + options.adaptStepSize()
				+ ":" + options.adaptMassMatrix() + ":" + options.allowFiniteDifferences()
				+ ":" + options.sliceWidth() + ":" + options.maximumSliceSteps()
				+ ":" + options.storeDraws();
		return new RunManifest(version, sampler.getClass().getName(), hash(modelIdentity),
				hash(optionText), seed, startedEpochMillis, elapsedNanoseconds);
	}
	public String libraryVersion() { return libraryVersion; }
	public String sampler() { return sampler; }
	public String modelHash() { return modelHash; }
	public String optionsHash() { return optionsHash; }
	public long seed() { return seed; }
	public long startedEpochMillis() { return startedEpochMillis; }
	public long elapsedNanoseconds() { return elapsedNanoseconds; }
	private static String hash(String value) {
		try {
			byte[] digest = MessageDigest.getInstance("SHA-256")
					.digest(value.getBytes(StandardCharsets.UTF_8));
			StringBuilder result = new StringBuilder(64);
			for (byte element : digest) result.append(String.format("%02x", element & 0xff));
			return result.toString();
		} catch (NoSuchAlgorithmException exception) {
			throw new IllegalStateException("SHA-256 is unavailable", exception);
		}
	}
}
