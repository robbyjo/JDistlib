/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Creates one independently mutable reversible-jump sampler per chain. */
@FunctionalInterface
public interface ReversibleJumpSamplerFactory {
	ReversibleJumpSampler create();
}
