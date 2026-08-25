/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Distribution object exposing its smallest enclosing support interval. */
public interface SupportedDistribution {
	double getLowerBound();
	double getUpperBound();
}
