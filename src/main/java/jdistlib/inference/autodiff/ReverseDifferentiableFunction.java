/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.autodiff;

/** Builds a scalar expression from parameter handles on a reverse-mode tape. */
@FunctionalInterface
public interface ReverseDifferentiableFunction {
	int evaluate(ReverseTape tape, int[] parameters);
}
