/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;

/** Finite, semi-infinite, and whole-line numerical integration examples. */
public final class NumericalIntegrationExamples {
	private NumericalIntegrationExamples() {}
	public static void main(String[] arguments) {
		IntegrationResult finite = Integrate.integrate(x -> Math.sin(x), 0, Math.PI);
		IntegrationResult positive = Integrate.integrate(x -> Math.exp(-x),
				0, Double.POSITIVE_INFINITY);
		IntegrationResult whole = Integrate.integrate(x -> Math.exp(-x*x),
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
		System.out.println("finite=" + finite.result + " positive=" + positive.result
				+ " whole=" + whole.result);
	}
}
