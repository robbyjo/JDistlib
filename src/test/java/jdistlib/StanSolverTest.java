/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.solver.AlgebraicSolver;
import jdistlib.inference.solver.DaeSolver;
import jdistlib.inference.solver.OdeSolver;

public class StanSolverTest {
	@Test public void algebraicNewtonSolverFindsCoupledRoot() {
		AlgebraicSolver.Result result = AlgebraicSolver.solve((state, parameters, data, residual) -> {
			residual[0] = state[0]*state[0] + state[1] - 3;
			residual[1] = state[0] + state[1]*state[1] - 3;
		}, new double[] {1.2, 1.2}, null, null, AlgebraicSolver.Options.defaults());
		assertEquals((-1 + Math.sqrt(13))/2, result.solution()[0], 1e-8);
		assertEquals(result.solution()[0], result.solution()[1], 1e-8);
		assertTrue(result.residualNorm() < 1e-9);
	}

	@Test public void adaptiveOdeMatchesExponentialDecay() {
		double[][] states = OdeSolver.integrate((time, state, parameters, data, derivative) ->
				derivative[0] = -parameters[0]*state[0], new double[] {2}, 0,
				new double[] {.25, 1, 2}, new double[] {.7}, null, OdeSolver.Options.defaults());
		assertEquals(2*Math.exp(-.7*.25), states[0][0], 2e-8);
		assertEquals(2*Math.exp(-.7*2), states[2][0], 2e-8);
	}

	@Test public void implicitDaeSolvesResidualForm() {
		double[] times = new double[100]; for (int i = 0; i < times.length; i++) times[i] = (i+1)/100.0;
		double[][] states = DaeSolver.integrate((time, state, derivative, parameters, data, residual) ->
				residual[0] = derivative[0] + parameters[0]*state[0],
				new double[] {1}, 0, times, new double[] {.5}, null,
				AlgebraicSolver.Options.defaults());
		assertEquals(Math.exp(-.5), states[99][0], 1.5e-3);
	}
}
