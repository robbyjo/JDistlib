/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

/** Java implementation of a forward-declared Stan function. */
@FunctionalInterface
public interface StanExternalFunction {
	/** Evaluates flattened arguments and returns values plus a complete Jacobian. */
	ExternalFunctionResult evaluate(double[][] arguments);
}
