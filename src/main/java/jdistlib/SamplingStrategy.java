/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Sampling algorithm currently selected by a numerical distribution. */
public enum SamplingStrategy {
	INVERSE_CDF,
	CERTIFIED_REJECTION,
	ADAPTIVE_LOG_CONCAVE_REJECTION,
	WALKER_ALIAS
}
