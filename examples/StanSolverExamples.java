/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.inference.solver.AlgebraicSolver;
import jdistlib.inference.solver.DaeSolver;
import jdistlib.inference.solver.OdeSolver;

/** Java-native algebraic, ODE, and DAE solver examples for Stan migrations. */
public final class StanSolverExamples {
	private StanSolverExamples() {}
	public static void main(String[] arguments) {
		AlgebraicSolver.Result root = AlgebraicSolver.solve((x, p, d, residual) ->
				residual[0] = x[0]*x[0] - p[0], new double[] {1},
				new double[] {2}, null, AlgebraicSolver.Options.defaults());
		System.out.println("sqrt(2) = " + root.solution()[0]);

		double[] times = {.25, .5, 1};
		double[][] ode = OdeSolver.integrate((t, y, p, d, derivative) ->
				derivative[0] = -p[0]*y[0], new double[] {1}, 0, times,
				new double[] {.8}, null, OdeSolver.Options.defaults());
		System.out.println("ODE final = " + Arrays.toString(ode[ode.length-1]));

		double[][] dae = DaeSolver.integrate((t, y, derivative, p, d, residual) ->
				residual[0] = derivative[0] + p[0]*y[0], new double[] {1}, 0,
				times, new double[] {.8}, null, AlgebraicSolver.Options.defaults());
		System.out.println("DAE final = " + Arrays.toString(dae[dae.length-1]));
	}
}
