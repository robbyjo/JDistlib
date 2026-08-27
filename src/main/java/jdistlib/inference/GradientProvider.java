/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Reports whether a differentiable target uses analytic rather than fallback gradients. */
public interface GradientProvider {
	boolean hasAnalyticGradient();
}
