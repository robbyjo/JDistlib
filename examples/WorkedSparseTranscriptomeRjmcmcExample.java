/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import jdistlib.accelerator.Compute;
import jdistlib.accelerator.ComputeSelection;
import jdistlib.accelerator.ComputeBackends;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.inference.GaussianSparseCoefficientProposal;
import jdistlib.inference.PortableSparseSubsetCheckpoint;
import jdistlib.inference.ResidualInformedSparseCandidateProposal;
import jdistlib.inference.SparseResidualProvider;
import jdistlib.inference.SparseSubsetCheckpoint;
import jdistlib.inference.SparseSubsetCheckpointIO;
import jdistlib.inference.SparseSubsetExport;
import jdistlib.inference.SparseSubsetLogJoint;
import jdistlib.inference.SparseSubsetResult;
import jdistlib.inference.SparseSubsetRjSampler;
import jdistlib.inference.SparseSubsetSamplingOptions;
import jdistlib.inference.SparseSubsetState;
import jdistlib.inference.SparseSubsetSummary;
import jdistlib.inference.SparseSubsetTarget;
import jdistlib.math.MathFunctions;
import jdistlib.rng.MersenneTwister;

/** Restartable sparse RJMCMC analysis of the public GSE93272 expression array study. */
public final class WorkedSparseTranscriptomeRjmcmcExample {
	private static final double MODEL_SIZE_MEAN = 3.0, COEFFICIENT_SD = 1.0;
	private static final double SCORE_TEMPERATURE = 0.10, UNIFORM_PROPOSAL_MIXTURE = 0.10;
	private WorkedSparseTranscriptomeRjmcmcExample() {}

	public static void main(String[] arguments) throws Exception {
		Arguments options = Arguments.parse(arguments);
		if (options.help) { usage(); return; }
		Path clinicalPath = options.data.resolve("clinical.tsv");
		Path expressionPath = options.data.resolve("expression.tsv.gz");
		if (!Files.isRegularFile(clinicalPath) || !Files.isRegularFile(expressionPath))
			throw new IllegalArgumentException("Prepared input is missing. Run examples/gse93272/prepare-gse93272.R; expected "
					+ clinicalPath + " and " + expressionPath);

		Clinical clinical = Clinical.read(clinicalPath);
		Expression expression = Expression.read(expressionPath, clinical.sampleIds);
		RandomInterceptModel model = new RandomInterceptModel(clinical, expression.values, options.maximumActive);
		SparseSubsetTarget target = new SparseSubsetTarget(
				new String[] {"intercept", "age_z", "female", "batch2", "rin_z", "log_sigma", "log_subject_sd"},
				expression.genes, options.maximumActive, model);
		Path chainDirectory = options.output.resolve(String.format(Locale.ROOT, "chain-%02d", options.chain));
		Files.createDirectories(chainDirectory); Path checkpointPath = chainDirectory.resolve("sparse-rj.checkpoint");

		try (ComputeSelection compute = ComputeBackends.select(options.compute);
				PreparedTransposeProduct product = compute.backend().prepareTransposeProduct(expression.values);
				ResidualInformedSparseCandidateProposal candidates = new ResidualInformedSparseCandidateProposal(
						product, model, SCORE_TEMPERATURE, UNIFORM_PROPOSAL_MIXTURE)) {
			String modelFingerprint = "gse93272-crp-das28-ri-v1/" + sha256(clinicalPath) + "/" + sha256(expressionPath)
					+ "/p" + expression.genes.length + "/k" + options.maximumActive;
			String optionsFingerprint = "sparse-rj-v1/warmup=" + options.warmup + "/thin=" + options.thinning
					+ "/score-temperature=" + SCORE_TEMPERATURE + "/uniform=" + UNIFORM_PROPOSAL_MIXTURE
					+ "/backend=" + compute.selectedBackend() + "/chain=" + options.chain;
			SparseSubsetRjSampler sampler = new SparseSubsetRjSampler(candidates,
					new GaussianSparseCoefficientProposal(0.0, COEFFICIENT_SD), 0.05, 0.30);
			SparseSubsetCheckpoint checkpoint = Files.isRegularFile(checkpointPath)
					? SparseSubsetCheckpointIO.read(checkpointPath, modelFingerprint, optionsFingerprint).checkpoint() : null;
			System.out.println("Data: " + clinical.sampleIds.length + " samples, " + clinical.subjectCount
					+ " subjects, " + expression.genes.length + " genes");
			System.out.println("Compute: " + compute.description());
			if (checkpoint != null) System.out.println("Resuming transition " + checkpoint.completedTransitions()
					+ " with " + checkpoint.retainedDraws() + " retained draws");

			for (int segment = 0; segment < options.segments; segment++) {
				final int currentSegment = segment;
				long firstRetained = checkpoint == null ? 0L : checkpoint.retainedDraws();
				SparseSubsetSamplingOptions sampling = SparseSubsetSamplingOptions.builder()
						.warmupIterations(options.warmup).segmentTransitions(options.segmentTransitions)
						.thinning(options.thinning).progressListener((done, segmentTotal, total, warmup, stats) -> {
							if (done == options.segmentTransitions || done % Math.max(1, options.segmentTransitions / 10) == 0)
								System.out.printf(Locale.ROOT, "segment %d/%d: %,d transitions (%s), size=%d%n",
										currentSegment + 1, options.segments, total, warmup ? "warmup" : "sampling", stats.toSize());
						}).build();
				SparseSubsetResult result = checkpoint == null
						? sampler.sample(target, target.state(new double[] {0, 0, 0, 0, 0, -0.35, -1.0}), sampling,
								new MersenneTwister(20260828L + options.chain))
						: sampler.resume(target, checkpoint, sampling);
				if (result.status() != SparseSubsetResult.Status.SUCCESS)
					throw new IllegalStateException("sparse RJ segment stopped with status " + result.status() + ": " + result.warnings());
				checkpoint = result.checkpoint();
				if (result.size() > 0) {
					Path draws = chainDirectory.resolve(String.format(Locale.ROOT, "draws-%012d-%012d.tsv",
							firstRetained, checkpoint.retainedDraws()));
					SparseSubsetExport.writeTidySegmentAtomic(draws, result, target, firstRetained);
				}
				// Commit the draw segment first. If power fails before this atomic checkpoint move,
				// replay starts from the old state and replaces the same deterministic segment file.
				writeAtomic(chainDirectory.resolve("summary.json"), new SparseSubsetSummary(target, checkpoint).toJson(0.001));
				SparseSubsetCheckpointIO.writeAtomic(checkpointPath, checkpoint, modelFingerprint, optionsFingerprint);
			}
			printTop(target, checkpoint, 20);
			System.out.println("Checkpoint: " + checkpointPath.toAbsolutePath());
		}
	}

	private static final class RandomInterceptModel implements SparseSubsetLogJoint, SparseResidualProvider {
		private final Clinical clinical; private final double[][] expression; private final int maximumActive;
		private final int[][] subjectRows; private final double logModelSizeNormalizer;
		RandomInterceptModel(Clinical clinical, double[][] expression, int maximumActive) {
			this.clinical = clinical; this.expression = expression; this.maximumActive = maximumActive;
			List<List<Integer>> rows = new ArrayList<List<Integer>>();
			for (int subject = 0; subject < clinical.subjectCount; subject++) rows.add(new ArrayList<Integer>());
			for (int row = 0; row < clinical.subject.length; row++) rows.get(clinical.subject[row]).add(Integer.valueOf(row));
			subjectRows = new int[rows.size()][];
			for (int subject = 0; subject < rows.size(); subject++) { subjectRows[subject] = new int[rows.get(subject).size()];
				for (int i = 0; i < subjectRows[subject].length; i++) subjectRows[subject][i] = rows.get(subject).get(i).intValue(); }
			double probability = 0.0; for (int size = 0; size <= maximumActive; size++)
				probability += Math.exp(size * Math.log(MODEL_SIZE_MEAN) - MODEL_SIZE_MEAN - MathFunctions.lgammafn(size + 1.0));
			logModelSizeNormalizer = Math.log(probability);
		}
		@Override public double logJoint(double[] common, int[] active, double[] coefficients) {
			double sigma = Math.exp(common[5]), subjectSd = Math.exp(common[6]);
			if (!(sigma >= 1e-4 && sigma <= 100.0 && subjectSd >= 1e-5 && subjectSd <= 100.0)) return Double.NEGATIVE_INFINITY;
			double[] residual = residual(common, active, coefficients); double sigma2 = sigma * sigma, subjectVariance = subjectSd * subjectSd;
			double logLikelihood = -0.5 * residual.length * Math.log(2.0 * Math.PI);
			for (int[] rows : subjectRows) { double sum = 0.0, sumSquares = 0.0; for (int row : rows) { sum += residual[row]; sumSquares += residual[row] * residual[row]; }
				double denominator = sigma2 + rows.length * subjectVariance;
				logLikelihood -= 0.5 * ((rows.length - 1.0) * Math.log(sigma2) + Math.log(denominator)
						+ sumSquares / sigma2 - subjectVariance * sum * sum / (sigma2 * denominator)); }
			double logPrior = 0.0; for (int i = 0; i < 5; i++) logPrior += normalLogDensity(common[i], 0.0, 2.5);
			logPrior += normalLogDensity(common[5], -0.5, 1.0) + normalLogDensity(common[6], -1.0, 1.0);
			for (double coefficient : coefficients) logPrior += normalLogDensity(coefficient, 0.0, COEFFICIENT_SD);
			int size = active.length; logPrior += size * Math.log(MODEL_SIZE_MEAN) - MODEL_SIZE_MEAN
					- MathFunctions.lgammafn(size + 1.0) - logModelSizeNormalizer - logChoose(expression[0].length, size);
			return logLikelihood + logPrior;
		}
		@Override public double[] scoreVector(SparseSubsetState state) {
			double[] common = state.commonParameters(), residual = residual(common, state.activeCandidates(), state.coefficients());
			double sigma2 = Math.exp(2.0 * common[5]), subjectVariance = Math.exp(2.0 * common[6]);
			for (int[] rows : subjectRows) { double sum = 0.0; for (int row : rows) sum += residual[row];
				double correction = subjectVariance * sum / (sigma2 * (sigma2 + rows.length * subjectVariance));
				for (int row : rows) residual[row] = residual[row] / sigma2 - correction; }
			return residual;
		}
		private double[] residual(double[] common, int[] active, double[] coefficients) {
			double[] residual = new double[clinical.outcome.length];
			for (int row = 0; row < residual.length; row++) { double mean = common[0] + common[1] * clinical.age[row]
					+ common[2] * clinical.female[row] + common[3] * clinical.batch2[row] + common[4] * clinical.rin[row];
				for (int coefficient = 0; coefficient < active.length; coefficient++) mean += coefficients[coefficient] * expression[row][active[coefficient]];
				residual[row] = clinical.outcome[row] - mean; }
			return residual;
		}
	}

	private static final class Clinical {
		final String[] sampleIds; final int[] subject; final int subjectCount; final double[] outcome, age, female, batch2, rin;
		Clinical(String[] sampleIds, int[] subject, int subjectCount, double[] outcome, double[] age, double[] female, double[] batch2, double[] rin) {
			this.sampleIds = sampleIds; this.subject = subject; this.subjectCount = subjectCount; this.outcome = outcome;
			this.age = age; this.female = female; this.batch2 = batch2; this.rin = rin;
		}
		static Clinical read(Path path) throws IOException {
			List<String[]> rows = new ArrayList<String[]>();
			try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
				String header = reader.readLine(); if (!"sample_id\tsubject_id\toutcome\tage_z\tfemale\tbatch2\trin_z".equals(header))
					throw new IOException("unexpected clinical.tsv header: " + header);
				String line; while ((line = reader.readLine()) != null) if (!line.isEmpty()) rows.add(line.split("\t", -1));
			}
			String[] ids = new String[rows.size()]; int[] subject = new int[rows.size()]; double[][] values = new double[5][rows.size()];
			Map<String, Integer> subjects = new LinkedHashMap<String, Integer>();
			for (int row = 0; row < rows.size(); row++) { String[] fields = rows.get(row); if (fields.length != 7) throw new IOException("invalid clinical row " + (row + 2));
				ids[row] = fields[0]; Integer index = subjects.get(fields[1]); if (index == null) { index = Integer.valueOf(subjects.size()); subjects.put(fields[1], index); }
				subject[row] = index.intValue(); for (int column = 0; column < values.length; column++) values[column][row] = Double.parseDouble(fields[column + 2]); }
			return new Clinical(ids, subject, subjects.size(), values[0], values[1], values[2], values[3], values[4]);
		}
	}

	private static final class Expression {
		final String[] genes; final double[][] values;
		Expression(String[] genes, double[][] values) { this.genes = genes; this.values = values; }
		static Expression read(Path path, String[] expectedSamples) throws IOException {
			List<String> genes = new ArrayList<String>(); List<double[]> rows = new ArrayList<double[]>();
			try (BufferedReader reader = reader(path)) {
				String[] header = reader.readLine().split("\t", -1); if (header.length != expectedSamples.length + 1 || !"gene".equals(header[0])) throw new IOException("invalid expression header");
				for (int sample = 0; sample < expectedSamples.length; sample++) if (!expectedSamples[sample].equals(header[sample + 1])) throw new IOException("expression sample order mismatch at " + sample);
				String line; while ((line = reader.readLine()) != null) if (!line.isEmpty()) { String[] fields = line.split("\t", -1);
					if (fields.length != header.length) throw new IOException("invalid expression row for " + fields[0]); genes.add(fields[0]); double[] value = new double[expectedSamples.length];
					for (int sample = 0; sample < value.length; sample++) value[sample] = Double.parseDouble(fields[sample + 1]); rows.add(value); }
			}
			double[][] values = new double[expectedSamples.length][rows.size()];
			for (int gene = 0; gene < rows.size(); gene++) for (int sample = 0; sample < expectedSamples.length; sample++) values[sample][gene] = rows.get(gene)[sample];
			return new Expression(genes.toArray(new String[genes.size()]), values);
		}
		private static BufferedReader reader(Path path) throws IOException { InputStream input = Files.newInputStream(path);
			if (path.getFileName().toString().endsWith(".gz")) input = new GZIPInputStream(input); return new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8)); }
	}

	private static final class Arguments {
		Path data, output = Paths.get("build", "example-output", "gse93272-sparse-rj"); Compute compute = Compute.AUTO;
		int chain = 1, segments = 1, segmentTransitions = 10000, thinning = 10, maximumActive = 20; long warmup = 50000L; boolean help;
		static Arguments parse(String[] values) { Arguments result = new Arguments();
			for (String value : values) { if ("--help".equals(value) || "-h".equals(value)) result.help = true;
				else if (value.startsWith("--data=")) result.data = Paths.get(value.substring(7)); else if (value.startsWith("--output=")) result.output = Paths.get(value.substring(9));
				else if (value.startsWith("--compute=")) result.compute = Compute.parse(value.substring(10)); else if (value.startsWith("--chain=")) result.chain = integer(value, 8);
				else if (value.startsWith("--segments=")) result.segments = integer(value, 11); else if (value.startsWith("--segment-transitions=")) result.segmentTransitions = integer(value, 22);
				else if (value.startsWith("--warmup=")) result.warmup = Long.parseLong(value.substring(9)); else if (value.startsWith("--thin=")) result.thinning = integer(value, 7);
				else if (value.startsWith("--max-active=")) result.maximumActive = integer(value, 13); else throw new IllegalArgumentException("unknown option " + value); }
			if (!result.help && result.data == null) throw new IllegalArgumentException("--data=<prepared-directory> is required (use --help)");
			if (result.chain < 1 || result.segments < 1 || result.segmentTransitions < 1 || result.thinning < 1 || result.maximumActive < 1 || result.warmup < 0L) throw new IllegalArgumentException("positive run options required");
			return result; }
		private static int integer(String value, int offset) { return Integer.parseInt(value.substring(offset)); }
	}

	private static void printTop(SparseSubsetTarget target, SparseSubsetCheckpoint checkpoint, int count) {
		SparseSubsetSummary summary = new SparseSubsetSummary(target, checkpoint); Integer[] indices = new Integer[target.candidateCount()];
		for (int i = 0; i < indices.length; i++) indices[i] = Integer.valueOf(i);
		Arrays.sort(indices, Comparator.comparingDouble((Integer index) -> summary.inclusionProbability(index.intValue())).reversed());
		System.out.println("Top posterior inclusion probabilities (exploratory; inspect convergence across chains):");
		for (int rank = 0; rank < Math.min(count, indices.length); rank++) { int index = indices[rank].intValue();
			System.out.printf(Locale.ROOT, "  %-20s PIP=%7.4f  beta|included=%8.4f%n", target.candidateName(index),
					summary.inclusionProbability(index), summary.conditionalCoefficientMean(index)); }
	}
	private static double normalLogDensity(double value, double mean, double sd) { double z = (value - mean) / sd; return -0.5 * z * z - Math.log(sd) - 0.5 * Math.log(2.0 * Math.PI); }
	private static double logChoose(int n, int k) { return MathFunctions.lgammafn(n + 1.0) - MathFunctions.lgammafn(k + 1.0) - MathFunctions.lgammafn(n - k + 1.0); }
	private static String sha256(Path path) throws Exception { MessageDigest digest = MessageDigest.getInstance("SHA-256"); byte[] buffer = new byte[1024 * 1024];
		try (InputStream input = Files.newInputStream(path)) { int read; while ((read = input.read(buffer)) >= 0) if (read > 0) digest.update(buffer, 0, read); }
		StringBuilder result = new StringBuilder(); for (byte value : digest.digest()) result.append(String.format(Locale.ROOT, "%02x", value & 255)); return result.toString(); }
	private static void writeAtomic(Path path, String text) throws IOException { Path temporary = path.resolveSibling(path.getFileName().toString() + ".tmp"); Files.write(temporary, text.getBytes(StandardCharsets.UTF_8));
		try { Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); } catch (AtomicMoveNotSupportedException exception) { Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING); } }
	private static void usage() { System.out.println("Usage: WorkedSparseTranscriptomeRjmcmcExample --data=DIR [--output=DIR] [--compute=auto|cpu|cuda|opencl|vulkan]\n"
			+ "       [--chain=1] [--segments=1] [--segment-transitions=10000] [--warmup=50000] [--thin=10] [--max-active=20]"); }
}
