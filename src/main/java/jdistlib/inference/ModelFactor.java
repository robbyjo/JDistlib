/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** One prior, likelihood, or constraint contribution to a model log density. */
@FunctionalInterface
public interface ModelFactor {
	double logDensity(ModelState state);
}
