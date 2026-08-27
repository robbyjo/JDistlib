/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

/** Two compilable Stan-inspired examples beyond the introductory coin model. */
public final class ModelScriptExamples {
	private ModelScriptExamples() {}
	public static void main(String[] arguments) {
		CompiledModelScript normal = normalLocationScale();
		BayesianModel model = normal.model();
		System.out.println(Gradients.check(model, model.initialState(), 1e-5, 1e-5));
		System.out.println(normal.generate(model.initialState(), new MersenneTwister(8)));

		CompiledModelScript counts = gammaPoisson();
		System.out.println(counts.generate(counts.model().initialState(),
				new MersenneTwister(9)));
	}
	public static CompiledModelScript normalLocationScale() {
		String source = "data { int N; vector[N] y; }\n"
				+ "parameters { real mu; real<lower=0> sigma; }\n"
				+ "model { mu ~ normal(0,10); sigma ~ exponential(1); "
				+ "y ~ normal(mu,sigma); }\n"
				+ "generated quantities { real y_rep = normal_rng(mu,sigma); }\n";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {5});
		data.put("y", new double[] {-0.2, 0.1, 0.4, 0.0, 0.3});
		return ModelScript.compile(source, data);
	}
	public static CompiledModelScript gammaPoisson() {
		String source = "data { int y; } parameters { real<lower=0> lambda; } "
				+ "model { lambda ~ gamma(2,1); y ~ poisson(lambda); } "
				+ "generated quantities { int y_rep = poisson_rng(lambda); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("y", new double[] {12});
		return ModelScript.compile(source, data);
	}
}
