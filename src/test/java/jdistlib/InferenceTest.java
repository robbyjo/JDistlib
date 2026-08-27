/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ChainExport;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.ChartSpec;
import jdistlib.inference.Constraints;
import jdistlib.inference.DiagnosticGraphs;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.InferenceHtmlReport;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.ModelGraphExport;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.RandomWalkMetropolis;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.LoadedGeneratedModel;
import jdistlib.inference.lang.ModelCompilationCache;
import jdistlib.inference.lang.ModelScript;
import jdistlib.inference.lang.ModelScriptException;
import jdistlib.inference.lang.ModelSourceGenerator;
import jdistlib.rng.MersenneTwister;

public class InferenceTest {
	@Test public void constrainedModelGradientIncludesJacobian() {
		BayesianModel model = new ModelBuilder()
				.parameter("scale", Constraints.positive(), 1.0)
				.factor("scale prior", new String[] {"scale"},
						ModelFactors.normalPrior("scale", 0.0, 2.0))
				.build();
		assertTrue(model.hasAnalyticGradient());
		GradientCheckResult check = Gradients.check(model, model.initialState(),
				2e-6, 2e-6);
		assertTrue(check.message(), check.passed());
		assertEquals(1.0, model.state(model.initialState()).scalar("scale"), 0.0);
	}

	@Test public void nutsSamplesNormalAndReportsHealth() {
		BayesianModel model = new ModelBuilder()
				.parameter("x", Constraints.real(), 0.0)
				.factor("x prior", new String[] {"x"},
						ModelFactors.normalPrior("x", 1.5, 0.75))
				.build();
		SamplingOptions options = SamplingOptions.builder()
				.warmupIterations(250).sampleIterations(350)
				.maximumTreeDepth(8).build();
		double[][] initial = {{-1.0}, {0.0}, {1.0}, {2.0}};
		ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), model,
				initial, options, 8675309L, 2);
		for (ChainResult chain : chains) assertEquals(ChainResult.Status.SUCCESS, chain.status());
		McmcDiagnosticReport report = McmcDiagnostics.analyze(new String[] {"x"}, chains);
		assertEquals(1.5, report.parameter("x").mean(), 0.12);
		assertTrue(report.parameter("x").bulkEffectiveSampleSize() > 100.0);
		assertEquals(0, report.sampler().divergences());
	}

	@Test public void scriptCompilesDifferentiatesSamplesAndGenerates() {
		String source = "data { int n; int y; }\n"
				+ "parameters { real<lower=0, upper=1> theta; }\n"
				+ "model { theta ~ beta(2, 2); y ~ binomial(n, theta); }\n"
				+ "generated quantities { int y_rep = binomial_rng(n, theta); }\n";
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("n", new double[] {10}); data.put("y", new double[] {7});
		CompiledModelScript compiled = ModelScript.compile(source, data);
		BayesianModel model = compiled.model();
		assertTrue(model.hasAnalyticGradient());
		assertTrue(Gradients.check(model, model.initialState(), 5e-6, 5e-6).passed());
		SamplingOptions options = SamplingOptions.builder().warmupIterations(200)
				.sampleIterations(500).maximumTreeDepth(8).build();
		ChainResult chain = new NoUTurnSampler().sample(model, model.initialState(),
				options, new MersenneTwister(1234L));
		double mean = 0.0;
		for (double[] state : chain.samples()) mean += model.state(state).scalar("theta");
		mean /= chain.size();
		assertEquals(9.0 / 14.0, mean, 0.08);
		double generated = compiled.generate(chain.sample(0), new MersenneTwister(7L))
				.get("y_rep")[0];
		assertTrue(generated >= 0.0 && generated <= 10.0);
	}

	@Test public void malformedScriptReportsSourceLocation() {
		try {
			ModelScript.validateSyntax("parameters { real x } model { x ~ normal(0, 1); }");
		} catch (ModelScriptException exception) {
			assertFalse(exception.diagnostics().isEmpty());
			assertTrue(exception.diagnostics().get(0).line() >= 1);
			return;
		}
		throw new AssertionError("malformed script was accepted");
	}

	@Test public void graphDataExportsWithoutUiDependency() {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(50)
				.sampleIterations(80).stepSize(0.5).build();
		ChainResult[] chains = Chains.parallel(new RandomWalkMetropolis(),
				x -> -0.5 * x[0] * x[0], new double[][] {{-1}, {1}},
				options, 42L, 2);
		ChartSpec trace = DiagnosticGraphs.trace("x", 0, chains);
		ChartSpec ranks = DiagnosticGraphs.ranks("x", 0, 10, chains);
		assertTrue(trace.toJson().contains("jdistlib.chart/1"));
		assertTrue(trace.toCsv().startsWith("series,x,y"));
		assertTrue(trace.toSvg(640, 320).startsWith("<svg"));
		assertTrue(ranks.toSvg(640, 320).contains("<rect"));
		McmcDiagnosticReport report = McmcDiagnostics.analyze(new String[] {"x"}, chains);
		assertTrue(ChainExport.toJson(new String[] {"x"}, chains).contains("jdistlib.chains/1"));
		assertTrue(ChainExport.toCsv(new String[] {"x"}, chains).contains("log_density"));
		assertTrue(InferenceHtmlReport.render("Test", report, null, trace)
				.contains("<!doctype html>"));
	}

	@Test public void modelGraphAndGeneratedSourceAreInspectable() throws Exception {
		BayesianModel model = new ModelBuilder().data("y", 1.0, 2.0)
				.parameter("mu", Constraints.real(), 0.0)
				.factor("likelihood", new String[] {"y", "mu"},
						ModelFactors.normalObservations("y", "mu", 1.0)).build();
		assertTrue(ModelGraphExport.toDot(model.graph()).contains("likelihood"));
		assertTrue(ModelGraphExport.toJson(model.graph()).contains("jdistlib.model-graph/1"));
		String script = "parameters { real x; } model { x ~ normal(0, 1); }";
		String generated = ModelSourceGenerator.generate("example.GeneratedNormal", script);
		assertTrue(generated.contains("implements GeneratedModelFactory"));
		Path cache = Files.createTempDirectory("jdistlib-model-cache-");
		try (LoadedGeneratedModel loaded = ModelCompilationCache.compile(script, cache)) {
			assertEquals(ModelSourceGenerator.hash(script), loaded.factory().sourceHash());
			assertEquals(1, loaded.factory().compile(new LinkedHashMap<String, double[]>())
					.model().dimension());
		}
	}
}
