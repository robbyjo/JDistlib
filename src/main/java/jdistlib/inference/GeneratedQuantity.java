/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Named scalar generated from one retained unconstrained state. */
public interface GeneratedQuantity {
	String name();
	double evaluate(double[] state);
}
