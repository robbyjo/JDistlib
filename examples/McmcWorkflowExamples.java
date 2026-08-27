/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.Constraints;
import jdistlib.inference.DiagnosticGraphs;
import jdistlib.inference.InferenceHtmlReport;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;

/** Equivalent Java-builder and Stan-inspired MCMC workflows with diagnostics. */
public final class McmcWorkflowExamples {
	private McmcWorkflowExamples() {}
	public static BayesianModel programmaticCoinModel() {
		return new ModelBuilder().data("n", 10).data("y", 7)
				.parameter("theta", Constraints.bounded(0, 1), 0.5)
				.factor("prior", new String[] {"theta"},
						ModelFactors.betaPrior("theta", 2, 2))
				.factor("likelihood", new String[] {"n", "y", "theta"},
						ModelFactors.binomialObservation("y", "n", "theta")).build();
	}
	public static CompiledModelScript scriptedCoinModel() {
		String source = "data { int n; int y; } "
				+ "parameters { real<lower=0,upper=1> theta; } "
				+ "model { theta ~ beta(2,2); y ~ binomial(n,theta); } "
				+ "generated quantities { int y_rep = binomial_rng(n,theta); }";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("n", new double[] {10}); data.put("y", new double[] {7});
		return ModelScript.compile(source, data);
	}
	public static void main(String[] arguments) {
		BayesianModel model = programmaticCoinModel();
		SamplingOptions options = SamplingOptions.builder().warmupIterations(100)
				.sampleIterations(150).targetAcceptance(0.85).build();
		ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), model,
				new double[][] {{-1}, {0}, {0.5}, {1}}, options, 20260827L, 4);
		McmcDiagnosticReport report = McmcDiagnostics.analyze(
				new String[] {"theta_free"}, chains);
		String html = InferenceHtmlReport.render("Coin posterior", report,
				model.graph(), DiagnosticGraphs.trace("theta_free", 0, chains),
				DiagnosticGraphs.ranks("theta_free", 0, 12, chains));
		System.out.println("reliable=" + report.reliable()
				+ " report characters=" + html.length()
				+ " script dimension=" + scriptedCoinModel().model().dimension());
	}
}
