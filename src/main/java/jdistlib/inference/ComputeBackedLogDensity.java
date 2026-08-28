/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.accelerator.ComputeBackend;

/** A differentiable target whose numerical evaluation is bound to a compute backend. */
public interface ComputeBackedLogDensity extends DifferentiableLogDensity {
	ComputeBackend computeBackend();
}
