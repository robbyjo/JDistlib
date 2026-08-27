/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.ChartSpec;
import jdistlib.inference.DiagnosticGraphs;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;

/** Compilable beta-binomial modeling, NUTS, diagnostics, and graph example. */
public final class InferenceExamples {
	private InferenceExamples() {}

	public static void main(String[] arguments) {
		String source = "data { int n; int y; }\n"
				+ "parameters { real<lower=0, upper=1> theta; }\n"
				+ "model { theta ~ beta(2, 2); y ~ binomial(n, theta); }\n"
				+ "generated quantities { int y_rep = binomial_rng(n, theta); }\n";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("n", new double[] {10});
		data.put("y", new double[] {7});
		CompiledModelScript compiled = ModelScript.compile(source, data);
		SamplingOptions options = SamplingOptions.builder()
				.warmupIterations(500).sampleIterations(1000)
				.targetAcceptance(0.85).build();
		double[][] initial = {{-1.0}, {-0.25}, {0.25}, {1.0}};
		ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), compiled.model(),
				initial, options, 20260826L, 4);
		McmcDiagnosticReport diagnostics = McmcDiagnostics.analyze(
				new String[] {"theta (unconstrained)"}, chains);
		ChartSpec trace = DiagnosticGraphs.trace("theta", 0, chains);
		System.out.println(diagnostics.toJson());
		System.out.println(trace.toSvg(800, 360));
	}
}
