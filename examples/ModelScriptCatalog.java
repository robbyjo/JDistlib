/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdistlib.inference.BayesianModel;
import jdistlib.inference.GradientCheckResult;
import jdistlib.inference.Gradients;
import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.ExternalFunctionRegistry;
import jdistlib.inference.lang.ExternalFunctionResult;
import jdistlib.inference.lang.ModelScript;
import jdistlib.rng.MersenneTwister;

/** Compiles and gradient-checks every example model script against representative data. */
public final class ModelScriptCatalog {
	private ModelScriptCatalog() {}
	public static void main(String[] arguments) throws IOException {
		List<Path> scripts = new ArrayList<Path>();
		List<Path> directories = new ArrayList<Path>();
		if (arguments.length == 0) {
			directories.add(Paths.get("examples/models"));
			directories.add(Paths.get("examples/stan"));
		} else {
			for (String argument : arguments) directories.add(Paths.get(argument));
		}
		for (Path directory : directories) {
			try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory,
					entry -> entry.toString().endsWith(".jdm") || entry.toString().endsWith(".stan"))) {
				for (Path path : stream) scripts.add(path);
			}
		}
		Collections.sort(scripts);
		if (scripts.size() < 30)
			throw new IllegalStateException("expected at least 30 model scripts, found " + scripts.size());
		for (Path path : scripts) validate(path);
		System.out.println("Validated " + scripts.size() + " JDistlib/Stan model scripts.");
	}

	private static void validate(Path path) throws IOException {
		String source = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
		CompiledModelScript compiled = path.toString().endsWith(".stan")
				? ModelScript.compileStan(source, data(path), externalFunctions(path))
				: ModelScript.compile(source, data(path));
		BayesianModel model = compiled.model();
		String name = path.getFileName().toString();
		double tolerance = name.matches("(?:38|39|40|41)-.*\\.stan") ? 3e-3 : 2e-5;
		GradientCheckResult gradient = Gradients.check(model, model.initialState(), tolerance, tolerance);
		if (!gradient.passed())
			throw new IllegalStateException(path + ": " + gradient);
		if (!Double.isFinite(model.logDensity(model.initialState())))
			throw new IllegalStateException(path + ": initial log density is not finite");
		compiled.generate(model.initialState(), new MersenneTwister(path.toString().hashCode()));
	}
	private static ExternalFunctionRegistry externalFunctions(Path path) {
		if (!path.getFileName().toString().startsWith("35-")) return ExternalFunctionRegistry.empty();
		return ExternalFunctionRegistry.builder().bind("java_penalty", arguments -> {
			double x = arguments[0][0], scale = arguments[1][0];
			return ExternalFunctionResult.scalar(scale*x*x, 2*scale*x, x*x);
		}).build();
	}

	private static Map<String, double[]> data(Path path) {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		String name = path.getFileName().toString();
		if (name.endsWith(".stan")) {
			switch (name.substring(0, 2)) {
			case "01": vectorData(result, new double[] {-0.3, 0.1, 0.4}); break;
			case "02": result.put("x", new double[] {1, -1, 1, 1});
				result.put("y", new double[] {-0.4, 0.7}); break;
			case "03": put(result, "y", 0.4); break;
			case "04": result.put("y", new double[] {-0.2, 0.1, 0.5}); break;
			case "05": break;
			case "06": result.put("x", new double[] {-1, 0, 1});
				result.put("y", new double[] {-0.8, 0.1, 1.2}); break;
			case "07": result.put("x", new double[] {10, 11, 12});
				result.put("y", new double[] {-0.8, 0.1, 1.2}); break;
			case "08": break;
			case "09": result.put("x", new double[] {1, 2, 3, 4, 5, 6}); break;
			case "10": result.put("X", new double[] {1, 0, 1, 1, 0, 1});
				result.put("y", new double[] {0.2, -0.1, 0.5});
				result.put("Sigma", new double[] {1.2, .2, .1, .2, 1.1, .15, .1, .15, .9}); break;
			case "11": result.put("y", new double[] {0.2, -0.1, 0.4}); break;
			case "12": break;
			case "13": case "14": case "15": case "16": case "17": case "18":
			case "19": case "20": case "21": case "22": case "23": case "24":
			case "25": case "26": case "27": case "28": case "29": case "30":
			case "31": case "32": case "33": case "34": case "35": case "36":
			case "37": case "38": case "39": case "40": case "41": break;
			default: throw new IllegalArgumentException("no representative data for " + name);
			}
			return result;
		}
		switch (name.substring(0, 2)) {
		case "01": put(result, "n", 10); put(result, "y", 7); break;
		case "02": put(result, "n_a", 100); put(result, "y_a", 12);
			put(result, "n_b", 100); put(result, "y_b", 18); break;
		case "03": put(result, "y", 1.2); break;
		case "04": vectorData(result, new double[] {-0.2, 0.1, 0.4, 0.0, 0.3}); break;
		case "05": vectorData(result, new double[] {-0.1, 0.2, 0.0, 15.0}); break;
		case "06": put(result, "y", 12); break;
		case "07": vectorData(result, new double[] {0.4, 1.2, 0.7, 2.1}); break;
		case "08": put(result, "y", 1); break;
		case "09": put(result, "n", 20); put(result, "y", 8); break;
		case "10": put(result, "y", 0.7); break;
		case "11": vectorData(result, new double[] {-0.4, 0.1, 0.3, 0.6}); break;
		case "12": put(result, "y", 2); put(result, "se", 1); break;
		case "13": put(result, "N", 4); result.put("log_y", new double[] {0.1, 0.4, -0.2, 0.3}); break;
		case "14": put(result, "x", 1.5); put(result, "y", 4); break;
		case "15": put(result, "x", 1); put(result, "y", 1); break;
		case "16": put(result, "x1", -1); put(result, "x2", 1);
			put(result, "y1", 0); put(result, "y2", 1); break;
		case "17": put(result, "N1", 3); put(result, "N2", 3);
			result.put("y1", new double[] {-0.2, 0.1, 0.0});
			result.put("y2", new double[] {0.7, 1.0, 1.2}); break;
		case "18": put(result, "y1", 2); put(result, "y2", -1);
			put(result, "se1", 1); put(result, "se2", 1.5); break;
		case "19": put(result, "y1", -2); put(result, "y2", 0); put(result, "y3", 2); break;
		case "20": put(result, "c1", 12); put(result, "c2", 7); put(result, "c3", 3); break;
		case "21": case "22": case "38": case "39": break;
		case "23": case "27": case "35": put(result, "y", 1); break;
		case "24": put(result, "y", 3); break;
		case "25": put(result, "y", 12); put(result, "offset", 10); break;
		case "26": put(result, "y", 1); break;
		case "28": put(result, "y", 1); break;
		case "29": put(result, "n", 10); put(result, "y", 4); break;
		case "30": put(result, "y", 5); break;
		case "31": put(result, "y", 2.5); break;
		case "32": put(result, "y", 0.8); break;
		case "33": put(result, "y", 0.2); break;
		case "34": put(result, "y", 4); break;
		case "36": put(result, "x", 0.5); put(result, "y", 1.2); break;
		case "37": result.put("y", new double[] {-1, 0, 1}); break;
		case "40": result.put("group_a", new double[] {-0.2, 0.1, 0.0});
			result.put("group_b", new double[] {0.7, 1.0, 1.2}); break;
		case "41": vectorData(result, new double[] {-0.42, -0.15, 0.03, 0.18,
				0.31, 0.44, 0.57, 0.66}); break;
		case "42": vectorData(result, new double[] {-0.3, 0.1, 0.2, 0.4, 6.0}); break;
		case "43": put(result, "x", 1.25); put(result, "y", 1); break;
		case "44": put(result, "n", 20); put(result, "y", 8); break;
		case "45": put(result, "x", 0.75); put(result, "y", 5); break;
		case "46": vectorData(result, new double[] {-0.4, 0.2, 0.5, 4.5});
			put(result, "cutoff", 2); break;
		case "47": put(result, "y", 4.5); break;
		case "48": put(result, "N", 4);
			result.put("lifetime", new double[] {0.7, 1.1, 1.8, 2.4}); break;
		case "49": put(result, "angle", 0.6); break;
		case "50": put(result, "N", 4);
			result.put("reaction_time", new double[] {0.8, 1.1, 1.6, 2.0}); break;
		default: throw new IllegalArgumentException("no representative data for " + name);
		}
		return result;
	}
	private static void vectorData(Map<String, double[]> data, double[] values) {
		put(data, "N", values.length); data.put("y", values);
	}
	private static void put(Map<String, double[]> data, String name, double value) {
		data.put(name, new double[] {value});
	}
}
