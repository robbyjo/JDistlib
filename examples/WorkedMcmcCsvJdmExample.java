/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.ChainExport;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.ChartSpec;
import jdistlib.inference.DiagnosticGraphs;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.InferenceHtmlReport;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.ParameterDiagnostics;
import jdistlib.inference.SamplerDiagnostics;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

/**
 * Fully worked CSV-to-JDM-to-MCMC example.
 *
 * <p>The default inputs are {@code examples/data/normal-observations.csv} and
 * {@code examples/models/41-normal-csv-mean.jdm}. Optional command-line
 * arguments replace the CSV path, JDM path, and output directory, in that
 * order.</p>
 */
public final class WorkedMcmcCsvJdmExample {
	/** A fixed base seed makes all four chains reproducible. */
	private static final long BASE_SEED = 2026082801L;

	private WorkedMcmcCsvJdmExample() {}

	public static void main(String[] arguments) throws IOException {
		/* Step 1: Resolve inputs. Keeping defaults in the repository makes this
		 * example runnable from the project root without additional setup. */
		Path csvPath = Paths.get(arguments.length > 0 ? arguments[0]
				: "examples/data/normal-observations.csv");
		Path modelPath = Paths.get(arguments.length > 1 ? arguments[1]
				: "examples/models/41-normal-csv-mean.jdm");
		Path outputDirectory = Paths.get(arguments.length > 2 ? arguments[2]
				: "build/worked-mcmc");
		Files.createDirectories(outputDirectory);

		/* Step 2: Read and validate the CSV column expected by the JDM model.
		 * Real applications may replace this deliberately small reader with their
		 * preferred CSV library; JDistlib only needs the final primitive array. */
		double[] observations = readNumericColumn(csvPath, "y");
		if (observations.length == 0)
			throw new IllegalArgumentException("the dataset must contain at least one observation");
		System.out.println("Loaded " + observations.length + " observations from " + csvPath);

		/* Step 3: Load the JDM model as UTF-8 text. The source remains ordinary
		 * repository data; no generated Java class is required for this workflow. */
		String modelSource = new String(Files.readAllBytes(modelPath), StandardCharsets.UTF_8);
		System.out.println("Loaded JDM source from " + modelPath);

		/* Step 4: Bind host data using names from the JDM data block.
		 * Scalars are represented by one-element arrays and vectors by double[]. */
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {observations.length});
		data.put("y", observations);

		/* Step 5: Compile the JDM source and obtain its Bayesian model.
		 * Compilation parses the script, validates dimensions/data, constructs
		 * constraints and factors, and prepares reusable reverse-mode gradients. */
		CompiledModelScript compiled = ModelScript.compile(modelSource, data);
		BayesianModel model = compiled.model();
		System.out.println("Compiled model with " + model.dimension() + " unconstrained parameter(s)");

		/* Step 6: Check the analytic/reverse gradient before invoking HMC/NUTS.
		 * A failed check usually indicates a model/kernel bug and should stop the run. */
		GradientCheckResult gradient = Gradients.check(
				model, model.initialState(), 2e-5, 2e-5);
		if (!gradient.passed()) throw new IllegalStateException(gradient.message());
		System.out.println("Gradient check: " + gradient.message());

		/* Step 7: Tune NUTS for a modest real analysis rather than accepting every
		 * default silently. Warmup adapts step size and the diagonal mass matrix;
		 * targetAcceptance=0.90 is conservative, and depth 11 leaves headroom for
		 * difficult trajectories while still bounding work. */
		SamplingOptions options = SamplingOptions.builder()
				.warmupIterations(600)
				.sampleIterations(1200)
				.thinning(1)
				.targetAcceptance(0.90)
				.maximumTreeDepth(11)
				.maximumEnergyError(1000.0)
				.adaptStepSize(true)
				.adaptMassMatrix(true)
				.denseMassMatrix(false)
				.build();
		System.out.println("NUTS options: warmup=" + options.warmupIterations()
				+ ", retained=" + options.sampleIterations()
				+ ", target acceptance=" + options.targetAcceptance()
				+ ", maximum depth=" + options.maximumTreeDepth());

		/* Step 8: Start four chains at dispersed values. For this model mu is an
		 * unconstrained real, so these one-coordinate arrays are also meaningful
		 * values on the scientific scale. */
		double[][] initialStates = {{-1.0}, {-0.25}, {0.25}, {1.0}};

		/* Step 9: Run deterministic parallel NUTS chains. The final argument is
		 * the worker count; scheduling does not change seeded results. */
		ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), model,
				initialStates, options, BASE_SEED, 4);

		/* Step 10: Check chain-level completion before pooling any draws. */
		for (int chain = 0; chain < chains.length; chain++) {
			System.out.println("Chain " + (chain + 1) + ": " + chains[chain].status()
					+ ", retained draws=" + chains[chain].size());
			for (String warning : chains[chain].warnings())
				System.out.println("  chain warning: " + warning);
			if (chains[chain].status() != ChainResult.Status.SUCCESS)
				throw new IllegalStateException("chain " + (chain + 1) + " did not complete");
		}

		/* Step 11: Compute rank-normalized R-hat, bulk/tail ESS, MCSE, acceptance,
		 * divergences, tree-depth saturation, numerical failures, and E-BFMI. */
		McmcDiagnosticReport report = McmcDiagnostics.analyze(new String[] {"mu"}, chains);
		ParameterDiagnostics mu = report.parameter("mu");
		SamplerDiagnostics sampler = report.sampler();

		/* Step 12: Print parameter summaries and convergence diagnostics together.
		 * Reading only a posterior mean without these checks is not sufficient. */
		System.out.println("\nPosterior summary for mu");
		System.out.printf("  mean = %.6f, sd = %.6f, median = %.6f%n",
				mu.mean(), mu.standardDeviation(), mu.median());
		System.out.printf("  95%% interval = [%.6f, %.6f]%n",
				mu.lowerQuantile(), mu.upperQuantile());
		System.out.printf("  R-hat = %.5f, bulk ESS = %.1f, tail ESS = %.1f, MCSE = %.6f%n",
				mu.rHat(), mu.bulkEffectiveSampleSize(), mu.tailEffectiveSampleSize(),
				mu.monteCarloStandardError());
		System.out.printf("  mean acceptance = %.4f, divergences = %d, depth saturations = %d%n",
				sampler.meanAcceptanceProbability(), sampler.divergences(),
				sampler.treeDepthSaturations());
		System.out.printf("  maximum observed depth = %d, E-BFMI = %.4f, numerical failures = %d%n",
				sampler.maximumTreeDepth(), sampler.energyBayesianFractionMissingInformation(),
				sampler.numericalFailures());
		for (String warning : report.warnings()) System.out.println("  diagnostic warning: " + warning);

		/* Step 13: Add a problem-specific posterior probability. Because mu is an
		 * unconstrained real, coordinate zero is already on its scientific scale. */
		int positive = 0, pooledDraws = 0;
		for (ChainResult chain : chains) for (int draw = 0; draw < chain.size(); draw++) {
			if (chain.valueAt(draw, 0) > 0) positive++;
			pooledDraws++;
		}
		double probabilityPositive = positive / (double) pooledDraws;
		System.out.printf("  Pr(mu > 0 | data) = %.4f%n", probabilityPositive);

		/* Step 14: This teaching model has a conjugate answer. Comparing MCMC with
		 * it checks the entire ingestion/compilation/sampling path, not just mixing. */
		double sum = 0.0;
		for (double observation : observations) sum += observation;
		double exactVariance = 1.0 / (observations.length + 1.0 / 100.0);
		double exactMean = exactVariance * sum;
		System.out.printf("  conjugate check: exact mean = %.6f, exact sd = %.6f, mean error = %.6f%n",
				exactMean, Math.sqrt(exactVariance), mu.mean() - exactMean);

		/* Step 15: Build renderer-neutral convergence plots. The trace plot is the
		 * requested chain-convergence view; rank, autocorrelation, and energy plots
		 * add complementary diagnostics. */
		ChartSpec trace = DiagnosticGraphs.trace("mu", 0, chains);
		ChartSpec ranks = DiagnosticGraphs.ranks("mu", 0, 20, chains);
		ChartSpec autocorrelation = DiagnosticGraphs.autocorrelation("mu", 0, 40, chains);
		ChartSpec energy = DiagnosticGraphs.energy(30, chains);

		/* Step 16: Save portable artifacts. SVG works in a browser, CSV supports
		 * downstream plotting, JSON is machine-readable, and the HTML report embeds
		 * all plots plus the model graph without a desktop UI dependency. */
		write(outputDirectory.resolve("mu-trace.svg"), trace.toSvg(1000, 420));
		write(outputDirectory.resolve("mu-ranks.svg"), ranks.toSvg(1000, 420));
		write(outputDirectory.resolve("mu-autocorrelation.svg"),
				autocorrelation.toSvg(1000, 420));
		write(outputDirectory.resolve("energy.svg"), energy.toSvg(1000, 420));
		write(outputDirectory.resolve("retained-draws.csv"),
				ChainExport.toCsv(new String[] {"mu"}, chains));
		write(outputDirectory.resolve("diagnostics.json"), report.toJson());
		write(outputDirectory.resolve("report.html"), InferenceHtmlReport.render(
				"Normal-mean CSV/JDM posterior", report, model.graph(),
				trace, ranks, autocorrelation, energy));

		/* Step 17: Exercise the JDM generated-quantities block using one retained
		 * unconstrained draw and an explicit RNG stream. */
		Map<String, double[]> generated = compiled.generate(
				chains[0].sample(chains[0].size() - 1), new MersenneTwister(BASE_SEED + 99));
		System.out.printf("  one posterior-predictive y_rep = %.6f%n", generated.get("y_rep")[0]);

		/* Step 18: State a cautious conclusion that separates computation from
		 * scientific evidence. A converged run can still support an inconclusive
		 * substantive answer. */
		System.out.println("\nConclusion");
		if (!report.reliable()) {
			System.out.println("  Do not interpret this posterior yet: at least one convergence or sampler diagnostic needs attention.");
		} else if (mu.lowerQuantile() > 0) {
			System.out.println("  Computation is healthy, and the 95% posterior interval supports a positive population mean.");
		} else if (mu.upperQuantile() < 0) {
			System.out.println("  Computation is healthy, and the 95% posterior interval supports a negative population mean.");
		} else {
			System.out.println("  Computation is healthy, but the 95% posterior interval crosses zero; the data do not establish the sign of the mean.");
		}
		System.out.println("  Review the convergence plot at "
				+ outputDirectory.resolve("mu-trace.svg").toAbsolutePath());
		System.out.println("  Review the complete report at "
				+ outputDirectory.resolve("report.html").toAbsolutePath());
	}

	/** Reads one numeric CSV column; quoted fields are intentionally out of scope. */
	private static double[] readNumericColumn(Path path, String requestedColumn)
			throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
			String header = reader.readLine();
			if (header == null) throw new IOException("CSV file is empty: " + path);
			String[] columns = header.split(",", -1);
			int selected = -1;
			for (int column = 0; column < columns.length; column++)
				if (columns[column].trim().equals(requestedColumn)) selected = column;
			if (selected < 0) throw new IOException("CSV column not found: " + requestedColumn);

			List<Double> values = new ArrayList<Double>();
			for (String line = reader.readLine(); line != null; line = reader.readLine()) {
				if (line.trim().isEmpty()) continue;
				String[] cells = line.split(",", -1);
				if (selected >= cells.length || cells[selected].trim().isEmpty())
					throw new IOException("missing " + requestedColumn + " on row " + (values.size() + 2));
				try { values.add(Double.valueOf(cells[selected].trim())); }
				catch (NumberFormatException exception) {
					throw new IOException("invalid number on row " + (values.size() + 2), exception);
				}
			}
			double[] result = new double[values.size()];
			for (int i = 0; i < result.length; i++) result[i] = values.get(i).doubleValue();
			return result;
		}
	}

	/** Writes a UTF-8 text artifact, replacing an earlier run's file. */
	private static void write(Path path, String content) throws IOException {
		Files.write(path, content.getBytes(StandardCharsets.UTF_8));
	}
}
