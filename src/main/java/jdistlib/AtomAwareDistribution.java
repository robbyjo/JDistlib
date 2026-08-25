/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** A scalar distribution that can report probability mass at an exact point. */
public interface AtomAwareDistribution {
	/** Returns {@code P(X = x)}; zero means no declared atom at {@code x}. */
	double atomProbability(double x);
}
