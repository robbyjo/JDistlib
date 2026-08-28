/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.inference.solver.AlgebraicSolver;
import jdistlib.inference.solver.DaeSolver;
import jdistlib.inference.solver.HigherIndexDaeSolver;
import jdistlib.inference.solver.HolonomicDaeSystem;
import jdistlib.inference.solver.OdeSolver;
import jdistlib.inference.solver.SensitivityResult;
import jdistlib.inference.solver.StiffOdeSolver;

/** Java-native algebraic, ODE, and DAE solver examples for Stan migrations. */
public final class StanSolverExamples {
	private StanSolverExamples() {}
	public static void main(String[] arguments) {
		SensitivityResult root = AlgebraicSolver.solveWithSensitivities((x, p, d, residual) ->
				residual[0] = x[0]*x[0] - p[0], new double[] {1},
				new double[] {2}, null, AlgebraicSolver.Options.defaults());
		System.out.println("sqrt(2) = " + root.values()[0][0]
				+ ", derivative = " + root.sensitivities()[0][0][0]);

		double[] times = {.25, .5, 1};
		SensitivityResult ode = OdeSolver.integrateWithSensitivities((t, y, p, d, derivative) ->
				derivative[0] = -p[0]*y[0], new double[] {1}, 0, times,
				new double[] {.8}, null, OdeSolver.Options.defaults());
		double[][] odeValues = ode.values();
		System.out.println("ODE final = " + Arrays.toString(odeValues[odeValues.length-1]));

		double[][] stiff = StiffOdeSolver.integrate((t, y, p, d, derivative) ->
				derivative[0] = -1000*(y[0]-Math.cos(t))-Math.sin(t),
				new double[] {1}, 0, times, null, null, StiffOdeSolver.Options.defaults());
		System.out.println("stiff ODE final = " + Arrays.toString(stiff[stiff.length-1]));

		double[][] dae = DaeSolver.integrate((t, y, derivative, p, d, residual) ->
				residual[0] = derivative[0] + p[0]*y[0], new double[] {1}, 0,
				times, new double[] {.8}, null, AlgebraicSolver.Options.defaults());
		System.out.println("DAE final = " + Arrays.toString(dae[dae.length-1]));

		HolonomicDaeSystem circle = new HolonomicDaeSystem() {
			@Override public int constraintCount() { return 1; }
			@Override public void acceleration(double t, double[] q, double[] v,
					double[] p, double[] d, double[] acceleration) {
				acceleration[0] = -q[0]; acceleration[1] = -q[1];
			}
			@Override public void constraints(double t, double[] q, double[] p,
					double[] d, double[] residual) {
				residual[0] = q[0]*q[0]+q[1]*q[1]-1;
			}
		};
		HigherIndexDaeSolver.Result pendulum = HigherIndexDaeSolver.integrate(circle,
				new double[] {1,0}, new double[] {0,1}, 0, times,
				.002, 1e-10, 20, null, null);
		System.out.println("index-3 position = "
				+ Arrays.toString(pendulum.positions()[times.length-1]));
	}
}
