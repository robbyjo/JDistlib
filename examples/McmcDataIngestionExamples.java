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
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.Constraints;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelFactors;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.SamplingOptions;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ModelScript;

/** Reads a CSV column and fits the same posterior through Java and script frontends. */
public final class McmcDataIngestionExamples {
	private McmcDataIngestionExamples() {}

	public static double[] readNumericColumn(Path path, String column) throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
			String header = reader.readLine();
			if (header == null) throw new IOException("CSV file is empty: " + path);
			String[] names = header.split(",", -1);
			int selected = -1;
			for (int i = 0; i < names.length; i++)
				if (names[i].trim().equals(column)) selected = i;
			if (selected < 0) throw new IOException("CSV column not found: " + column);
			List<Double> values = new ArrayList<Double>();
			for (String line = reader.readLine(); line != null; line = reader.readLine()) {
				if (line.trim().isEmpty()) continue;
				String[] cells = line.split(",", -1);
				if (selected >= cells.length || cells[selected].trim().isEmpty())
					throw new IOException("missing " + column + " value on CSV row " + (values.size() + 2));
				try { values.add(Double.valueOf(cells[selected].trim())); }
				catch (NumberFormatException exception) {
					throw new IOException("invalid numeric " + column + " value on CSV row "
							+ (values.size() + 2), exception);
				}
			}
			double[] result = new double[values.size()];
			for (int i = 0; i < result.length; i++) result[i] = values.get(i).doubleValue();
			return result;
		}
	}

	public static BayesianModel buildInJava(double[] observations) {
		return new ModelBuilder().data("y", observations)
				.parameter("mu", Constraints.real(), 0.0)
				.factor("mu prior", new String[] {"mu"},
						ModelFactors.normalPrior("mu", 0.0, 10.0))
				.factor("observations", new String[] {"y", "mu"},
						ModelFactors.normalObservations("y", "mu", 1.0)).build();
	}

	public static CompiledModelScript compileScript(Path script, double[] observations)
			throws IOException {
		String source = new String(Files.readAllBytes(script), StandardCharsets.UTF_8);
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		data.put("N", new double[] {observations.length});
		data.put("y", observations);
		return ModelScript.compile(source, data);
	}

	private static McmcDiagnosticReport sample(BayesianModel model, long seed) {
		SamplingOptions options = SamplingOptions.builder().warmupIterations(500)
				.sampleIterations(1000).targetAcceptance(0.85).build();
		ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), model,
				new double[][] {{-1.0}, {-0.25}, {0.25}, {1.0}}, options, seed, 4);
		return McmcDiagnostics.analyze(new String[] {"mu"}, chains);
	}

	public static void main(String[] arguments) throws IOException {
		Path csv = Paths.get(arguments.length > 0 ? arguments[0]
				: "examples/data/normal-observations.csv");
		Path script = Paths.get(arguments.length > 1 ? arguments[1]
				: "examples/models/41-normal-csv-mean.jdm");
		double[] observations = readNumericColumn(csv, "y");
		McmcDiagnosticReport javaReport = sample(buildInJava(observations), 2026082701L);
		McmcDiagnosticReport scriptReport = sample(
				compileScript(script, observations).model(), 2026082701L);
		System.out.println("Java mu mean: " + javaReport.parameter("mu").mean());
		System.out.println("Script mu mean: " + scriptReport.parameter("mu").mean());
	}
}
