/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.inference.lang.CompiledModelScript;
import jdistlib.inference.lang.LoadedGeneratedModel;
import jdistlib.inference.lang.ModelCompilationCache;
import jdistlib.inference.lang.ModelScript;
import jdistlib.inference.lang.ModelSourceGenerator;

/** Compares in-memory, cached-JDK, and generated-source script compilation. */
public final class ModelScriptCompilationExamples {
	private ModelScriptCompilationExamples() {}

	public static void main(String[] arguments) throws Exception {
		Path scriptPath = Paths.get(arguments.length > 0 ? arguments[0]
				: "examples/models/41-normal-csv-mean.jdm");
		Path cachePath = Paths.get(arguments.length > 1 ? arguments[1]
				: "build/model-script-cache");
		String source = new String(Files.readAllBytes(scriptPath), StandardCharsets.UTF_8);
		Map<String, double[]> data = normalData();

		CompiledModelScript inMemory = ModelScript.compile(source, data);
		String generatedJava = ModelSourceGenerator.generate(
				"example.generated.NormalCsvMean", source);
		System.out.println("Generated Java source characters: " + generatedJava.length());

		try (LoadedGeneratedModel cached = ModelCompilationCache.compile(source, cachePath)) {
			CompiledModelScript fromCache = cached.factory().compile(data);
			double[] state = inMemory.model().initialState();
			double directDensity = inMemory.model().logDensity(state);
			double cachedDensity = fromCache.model().logDensity(state);
			if (Double.doubleToLongBits(directDensity) != Double.doubleToLongBits(cachedDensity))
				throw new IllegalStateException("compilation paths changed model semantics");
			System.out.println("Source SHA-256: " + cached.factory().sourceHash());
			System.out.println("Initial log density: " + directDensity);
		}
	}

	private static Map<String, double[]> normalData() {
		Map<String, double[]> data = new LinkedHashMap<String, double[]>();
		double[] y = {-0.42, -0.15, 0.03, 0.18, 0.31, 0.44, 0.57, 0.66};
		data.put("N", new double[] {y.length});
		data.put("y", y);
		return data;
	}
}
