/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import jdistlib.inference.solver.AlgebraicSolver;
import jdistlib.inference.solver.DaeSolver;
import jdistlib.inference.solver.HigherIndexDaeSolver;
import jdistlib.inference.solver.HolonomicDaeSystem;
import jdistlib.inference.solver.OdeSolver;
import jdistlib.inference.solver.SensitivityResult;
import jdistlib.inference.solver.StiffOdeSolver;

public class StanSolverCompletionTest {
	@Test public void algebraicImplicitSensitivityMatchesClosedForm() {
		SensitivityResult result = AlgebraicSolver.solveWithSensitivities(
				(x, parameters, data, residual) -> residual[0] = x[0]*x[0]-parameters[0],
				new double[] {1}, new double[] {2}, null, AlgebraicSolver.Options.defaults());
		assertEquals(Math.sqrt(2), result.values()[0][0], 1e-9);
		assertEquals(1/(2*Math.sqrt(2)), result.sensitivities()[0][0][0], 2e-6);
	}

	@Test public void odeForwardSensitivitiesMatchReferenceCorpus() throws IOException {
		double[][] reference = corpus("decay-sensitivity.csv");
		double[] times = column(reference, 0);
		SensitivityResult result = OdeSolver.integrateWithSensitivities(
				(t, y, parameters, data, derivative) -> derivative[0] = -parameters[0]*y[0],
				new double[] {1}, 0, times, new double[] {.7}, null, OdeSolver.Options.defaults());
		for (int i = 0; i < times.length; i++) {
			assertEquals(reference[i][1], result.values()[i][0], 3e-8);
			assertEquals(reference[i][2], result.sensitivities()[i][0][0], 2e-6);
		}
	}

	@Test public void daeTrajectorySensitivityTracksDecayRate() {
		double[] times = new double[200]; for (int i = 0; i < times.length; i++) times[i] = (i+1)/200.0;
		SensitivityResult result = DaeSolver.integrateWithSensitivities(
				(t, y, derivative, parameters, data, residual) -> residual[0] = derivative[0]+parameters[0]*y[0],
				new double[] {1}, 0, times, new double[] {.5}, null, AlgebraicSolver.Options.defaults());
		assertEquals(-Math.exp(-.5), result.sensitivities()[199][0][0], 5e-3);
	}

	@Test public void stiffBdfPathMatchesIndependentReferenceCorpus() throws IOException {
		double[][] reference = corpus("stiff-linear.csv"); double[] times = column(reference, 0);
		double[][] result = StiffOdeSolver.integrate((t, y, parameters, data, derivative) ->
				derivative[0] = -1000*(y[0]-Math.cos(t))-Math.sin(t),
				new double[] {1}, 0, times, null, null, StiffOdeSolver.Options.defaults());
		for (int i = 0; i < times.length; i++) assertEquals(reference[i][1], result[i][0], 2e-5);
	}

	@Test public void projectedIndexThreePendulumMatchesIndependentReferenceCorpus() throws IOException {
		HolonomicDaeSystem pendulum = new HolonomicDaeSystem() {
			@Override public int constraintCount() { return 1; }
			@Override public void acceleration(double time, double[] q, double[] v,
					double[] parameters, double[] data, double[] acceleration) {
				acceleration[0] = -q[0]; acceleration[1] = -q[1];
			}
			@Override public void constraints(double time, double[] q, double[] parameters,
					double[] data, double[] residual) { residual[0] = q[0]*q[0]+q[1]*q[1]-1; }
		};
		double[][] reference = corpus("pendulum-index3.csv"); double[] times = column(reference, 0);
		HigherIndexDaeSolver.Result result = HigherIndexDaeSolver.integrate(pendulum,
				new double[] {1,0}, new double[] {0,1}, 0, times,
				.002, 1e-10, 20, null, null);
		double[][] q = result.positions(), v = result.velocities();
		for (int i = 0; i < times.length; i++) {
			assertEquals(reference[i][1], q[i][0], 2e-3); assertEquals(reference[i][2], q[i][1], 2e-3);
			assertEquals(reference[i][3], v[i][0], 2e-3); assertEquals(reference[i][4], v[i][1], 2e-3);
		}
		assertTrue(result.maximumConstraintError() < 2e-5);
	}

	private static double[][] corpus(String name) throws IOException {
		List<double[]> rows = new ArrayList<double[]>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(
				StanSolverCompletionTest.class.getResourceAsStream("/solver-reference/"+name),
				StandardCharsets.UTF_8))) {
			reader.readLine(); String line;
			while ((line = reader.readLine()) != null) {
				String[] fields = line.split(","); double[] values = new double[fields.length];
				for (int i = 0; i < fields.length; i++) values[i] = Double.parseDouble(fields[i]);
				rows.add(values);
			}
		}
		return rows.toArray(new double[rows.size()][]);
	}
	private static double[] column(double[][] values, int column) {
		double[] result = new double[values.length]; for (int i = 0; i < result.length; i++) result[i] = values[i][column]; return result;
	}
}
