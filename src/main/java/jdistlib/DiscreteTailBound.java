/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/**
 * User-supplied certificate bounding all unnormalized mass beginning at an
 * omitted integer. The returned bound must include {@code firstOmittedWeight}.
 */
@FunctionalInterface
public interface DiscreteTailBound {
	double upperBound(long firstOmitted, double firstOmittedWeight);
}
