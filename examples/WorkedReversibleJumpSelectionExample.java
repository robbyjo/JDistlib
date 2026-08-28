/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

import jdistlib.inference.GaussianRjBirthProposal;
import jdistlib.inference.PortableReversibleJumpCheckpoint;
import jdistlib.inference.ReversibleJumpCheckpointIO;
import jdistlib.inference.ReversibleJumpDiagnosticReport;
import jdistlib.inference.ReversibleJumpDiagnostics;
import jdistlib.inference.ReversibleJumpExport;
import jdistlib.inference.ReversibleJumpParameterSummary;
import jdistlib.inference.ReversibleJumpResult;
import jdistlib.inference.ReversibleJumpSampler;
import jdistlib.inference.ReversibleJumpSamplingOptions;
import jdistlib.inference.ReversibleJumpState;
import jdistlib.inference.SubsetSelectionRj;
import jdistlib.inference.SubsetSelectionTarget;
import jdistlib.rng.MersenneTwister;

/** Complete normalized linear-regression subset-selection RJMCMC analysis. */
public final class WorkedReversibleJumpSelectionExample {
	private static final double OBSERVATION_SD = 0.75;
	private static final double INTERCEPT_SD = 5.0;
	private static final double COEFFICIENT_SD = 2.0;
	private static final double INCLUSION_PROBABILITY = 0.3;
	private static final String MODEL_FINGERPRINT = "worked-rj-linear-v1";
	private static final String OPTIONS_FINGERPRINT = "warmup-1500-samples-4000-v1";

	private WorkedReversibleJumpSelectionExample() {}

	public static void main(String[] arguments) throws Exception {
		// The first two covariates generated the response. "proxy_marker" is correlated
		// with genotype, creating a realistic competing-locus model for swap moves.
		final double[][] x = {
			{-1.50, -1, -0.80}, {-1.30, 1, 0.90}, {-1.10, -1, -1.20}, {-0.90, 1, 0.70},
			{-0.70, -1, -0.60}, {-0.50, 1, 1.30}, {-0.30, -1, -1.10}, {-0.10, 1, 0.80},
			{0.10, -1, -0.90}, {0.30, 1, 1.20}, {0.50, -1, -0.70}, {0.70, 1, 0.60},
			{0.90, -1, -1.30}, {1.10, 1, 1.10}, {1.30, -1, -0.80}, {1.50, 1, 0.90}
		};
		double[] residuals = {0.10, -0.18, 0.24, -0.05, -0.20, 0.12, 0.04, -0.16,
				0.19, -0.08, 0.05, 0.15, -0.14, 0.09, -0.03, -0.11};
		final double[] y = new double[x.length];
		for (int row = 0; row < y.length; row++) y[row] = 1.1 + 1.4 * x[row][0] + 0.45 * x[row][1] + residuals[row];

		// Parameter order is alpha followed by coefficients for active candidates in name order.
		// Crucially, this callback returns a COMPLETE normalized log joint. Model-dependent
		// constants may not be discarded in a trans-dimensional calculation.
		SubsetSelectionTarget target = new SubsetSelectionTarget(new String[] {"alpha"},
				new String[] {"dose", "genotype", "proxy_marker"}, (common, active, coefficients) -> {
			double result = normalLogDensity(common[0], 0.0, INTERCEPT_SD);
			int included = active.length;
			result += included * Math.log(INCLUSION_PROBABILITY)
					+ (3 - included) * Math.log(1.0 - INCLUSION_PROBABILITY);
			for (double coefficient : coefficients) result += normalLogDensity(coefficient, 0.0, COEFFICIENT_SD);
			for (int row = 0; row < y.length; row++) {
				double mean = common[0];
				for (int coefficient = 0; coefficient < active.length; coefficient++)
					mean += coefficients[coefficient] * x[row][active[coefficient]];
				result += normalLogDensity(y[row], mean, OBSERVATION_SD);
			}
			return result;
		});

		// Drawing a new coefficient from its N(0, 2) prior makes the birth/death ratio
		// stable. Add/drop/swap selection probabilities and the unit insertion Jacobian
		// are included automatically by the subset-selection moves.
		ReversibleJumpSamplingOptions options = ReversibleJumpSamplingOptions.builder()
				.warmupIterations(1500).sampleIterations(4000).thinning(1)
				.targetJumpAcceptance(0.25).adaptMoveWeights(true).build();

		long[] initialModels = {0L, 1L, 2L, 4L};
		ReversibleJumpResult[] chains = new ReversibleJumpResult[initialModels.length];
		for (int chain = 0; chain < chains.length; chain++) {
			long model = initialModels[chain];
			ReversibleJumpState initial = target.state(model, new double[] {0.0}, new double[Long.bitCount(model)]);
			// Samplers own mutable warmup state, so each independent chain gets its own instance.
			ReversibleJumpSampler sampler = newSampler();
			chains[chain] = sampler.sample(target, initial, options, new MersenneTwister(20260829 + chain));
			if (chains[chain].status() != ReversibleJumpResult.Status.SUCCESS)
				throw new IllegalStateException("chain " + chain + " failed: " + chains[chain].status());
		}

		ReversibleJumpDiagnosticReport report = ReversibleJumpDiagnostics.analyze(target, chains);
		System.out.println("Posterior model probabilities:");
		long[] modelIds = report.modelIds(); double[] modelProbabilities = report.modelProbabilities();
		for (int model = 0; model < Math.min(8, modelIds.length); model++)
			System.out.printf(Locale.ROOT, "  %-25s %.4f%n", target.modelName(modelIds[model]), modelProbabilities[model]);

		System.out.println("\nPosterior inclusion probabilities:");
		String[] candidates = report.candidateNames(); double[] inclusion = report.inclusionProbabilities();
		double[] ess = report.inclusionEffectiveSampleSizes(), rhat = report.inclusionRHats();
		for (int candidate = 0; candidate < candidates.length; candidate++)
			System.out.printf(Locale.ROOT, "  %-12s probability=%6.3f  ESS=%8.1f  R-hat=%6.3f%n",
					candidates[candidate], inclusion[candidate], ess[candidate], rhat[candidate]);

		System.out.println("\nMove acceptance:");
		for (int move = 0; move < report.moveCount(); move++)
			System.out.printf(Locale.ROOT, "  %-8s %6.3f (%d/%d)%n", report.moveName(move),
					report.moveAcceptanceRate(move), report.moveAccepts(move), report.moveAttempts(move));

		System.out.println("\nParameter summaries conditional on inclusion:");
		for (ReversibleJumpParameterSummary summary : report.parameterSummaries())
			System.out.printf(Locale.ROOT, "  %-12s mean=%7.3f  sd=%7.3f  present=%d%n",
					summary.name(), summary.mean(), summary.standardDeviation(), summary.draws());

		Path output = arguments.length == 0 ? Paths.get("build", "example-output", "rjmcmc") : Paths.get(arguments[0]);
		Files.createDirectories(output);
		Files.write(output.resolve("rjmcmc-chain-1.csv"), ReversibleJumpExport.toTidyCsv(chains[0], target).getBytes(StandardCharsets.UTF_8));
		Files.write(output.resolve("rjmcmc-diagnostics.json"), report.toJson().getBytes(StandardCharsets.UTF_8));
		Path checkpointPath = output.resolve("rjmcmc.checkpoint");
		ReversibleJumpCheckpointIO.write(checkpointPath, chains[0].checkpoint(), MODEL_FINGERPRINT, OPTIONS_FINGERPRINT);

		// Loading a checkpoint restores the ragged state, frozen proposal adaptation,
		// move weights, completed iteration, and exact random stream.
		PortableReversibleJumpCheckpoint restored = ReversibleJumpCheckpointIO.read(
				checkpointPath, MODEL_FINGERPRINT, OPTIONS_FINGERPRINT);
		ReversibleJumpResult continuation = newSampler().resume(target,
				restored.checkpoint(), ReversibleJumpSamplingOptions.builder().warmupIterations(0)
						.sampleIterations(250).adaptMoveWeights(false).build());
		System.out.println("\nWrote output to " + output.toAbsolutePath());
		System.out.println("Deterministic continuation retained " + continuation.size() + " additional draws.");
		if (!report.warnings().isEmpty()) System.out.println("Diagnostic warnings: " + report.warnings());
	}

	private static ReversibleJumpSampler newSampler() {
		// The birth proposal adapts during warmup, so it must not be shared by independent chains.
		return SubsetSelectionRj.sampler(new GaussianRjBirthProposal(0.0, COEFFICIENT_SD), 0.20, 0.30);
	}

	private static double normalLogDensity(double value, double mean, double standardDeviation) {
		double standardized = (value - mean) / standardDeviation;
		return -0.5 * standardized * standardized - Math.log(standardDeviation) - 0.5 * Math.log(2.0 * Math.PI);
	}
}
