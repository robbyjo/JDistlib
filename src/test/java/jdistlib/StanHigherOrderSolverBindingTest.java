/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.ModelScript;

public class StanHigherOrderSolverBindingTest {
	@Test(expected=IllegalArgumentException.class)
	public void solverTimeGridMustBeDataOnly() {
		String source = "functions { vector decay(real t, vector y, vector theta, array[] real xr, array[] int xi) "
				+ "{ return [-theta[1]*y[1]]'; } } transformed data { array[1] real ts={.2}; "
				+ "array[1] real xr={0}; array[1] int xi={0}; } parameters { real t0; vector[1] theta; } "
				+ "model { array[1] vector[1] path=ode_rk45(decay,[1]',t0,ts,theta,xr,xi); target += path[1,1]; }";
		ModelScript.compileStan(source);
	}

	@Test public void oneDimensionalIntegratorBindsFunctionAndParameterSensitivities() {
		String source = "functions { real kernel(real x, real xc, array[] real theta, "
				+ "array[] real xr, array[] int xi) { return xr[1]*exp(-theta[1]*x); } } "
				+ "transformed data { array[1] real xr={2}; array[1] int xi={0}; } "
				+ "parameters { array[1] real<lower=.1> theta; } transformed parameters { "
				+ "real area = integrate_1d(kernel,0,1,theta,xr,xi,1e-8); } "
				+ "model { theta ~ lognormal(0,1); target += normal_lpdf(area | 1.2,.2); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, model.initialState(), 2e-4, 2e-4).passed());
	}

	@Test public void algebraSolverFunctionReferenceDifferentiatesImplicitRoot() {
		String source = "functions { vector equation(vector y, vector theta, array[] real xr, array[] int xi) { "
				+ "return [square(y[1])-theta[1]]'; } } transformed data { array[1] real xr={0}; array[1] int xi={0}; } "
				+ "parameters { vector<lower=0>[1] theta; } transformed parameters { vector[1] root = "
				+ "algebra_solver_newton(equation,[1]',theta,xr,xi); } model { theta ~ lognormal(0,1); "
				+ "target += normal_lpdf(root[1] | 1.5,.2); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, model.initialState(), 3e-4, 3e-4).passed());
		assertTrue(Double.isFinite(model.logDensity(model.initialState())));
	}

	@Test public void rk45AndBdfBindingsReturnDifferentiableArrayOfVectors() {
		String functions = "functions { vector decay(real t, vector y, vector theta, array[] real xr, array[] int xi) { "
				+ "return [-theta[1]*y[1]]'; } } transformed data { array[2] real ts={.25,.5}; "
				+ "array[1] real xr={0}; array[1] int xi={0}; } parameters { vector<lower=0>[1] theta; } ";
		for (String solver : new String[] {"ode_rk45", "ode_bdf"}) {
			String source = functions + "transformed parameters { array[2] vector[1] trajectory = "
					+ solver+"(decay,[1]',0,ts,theta,xr,xi); } model { theta ~ lognormal(0,1); "
					+ "target += normal_lpdf(trajectory[2,1] | .7,.1); }";
			BayesianModel model = ModelScript.compileStan(source).model();
			GradientCheckResult check = Gradients.check(model, model.initialState(), 2e-3, 2e-3);
			assertTrue(solver+": "+check.message(), check.passed());
		}
	}

	@Test public void daeBindingChecksInitialConsistencyAndDifferentiates() {
		String source = "functions { vector decay_residual(real t, vector y, vector yd, vector theta, "
				+ "array[] real xr, array[] int xi) { return [yd[1]+theta[1]*y[1]]'; } } "
				+ "transformed data { array[2] real ts={.1,.2}; array[1] real xr={0}; array[1] int xi={0}; } "
				+ "parameters { vector<lower=0>[1] theta; } transformed parameters { array[2] vector[1] trajectory = "
				+ "dae(decay_residual,[1]',[-theta[1]]',0,ts,theta,xr,xi); } model { theta ~ lognormal(0,1); "
				+ "target += normal_lpdf(trajectory[2,1] | .8,.1); }";
		BayesianModel model = ModelScript.compileStan(source).model();
		assertTrue(Gradients.check(model, model.initialState(), 1e-3, 1e-3).passed());
	}
}
