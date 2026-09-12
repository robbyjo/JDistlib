/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Source-generation CLI with an explicit {@code --run} mode for inference. */
public final class ModelScriptCli {
	private ModelScriptCli() {}
	public static void main(String[] arguments) throws Exception {
		for (String argument : arguments) if ("--run".equals(argument)) {
			ModelRunnerCli.main(arguments); return;
		}
		if (arguments.length == 1 && ("--help".equals(arguments[0]) || "-h".equals(arguments[0]))) {
			System.out.println("Source generation (existing mode): ModelScriptCli <model.jdm-or-stan> <fully.qualified.Class> <output.java>");
			System.out.println("Inference: java -jar jdistlib-all.jar --run --input script.stan --output output.txt [options]");
			System.out.println("Use --run --help for data bindings and sampler options.");
			return;
		}
		if (arguments.length != 3) {
			System.err.println("usage: ModelScriptCli <model.jdm> <fully.qualified.Class> <output.java>");
			System.exit(2); return;
		}
		Path input = Paths.get(arguments[0]);
		Path output = Paths.get(arguments[2]);
		String script = new String(Files.readAllBytes(input), StandardCharsets.UTF_8);
		ModelScript.validateSyntax(script);
		Path parent = output.toAbsolutePath().getParent();
		if (parent != null) Files.createDirectories(parent);
		Files.write(output, ModelSourceGenerator.generate(arguments[1], script)
				.getBytes(StandardCharsets.UTF_8));
	}
}
