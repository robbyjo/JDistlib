/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Sampler capable of restoring algorithm-specific adaptive state. */
public interface ResumableSampler extends Sampler {
	ChainResult resume(LogDensity target, ChainCheckpoint checkpoint,
			SamplingOptions options);
}
