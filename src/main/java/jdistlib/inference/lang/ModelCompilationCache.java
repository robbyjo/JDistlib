/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/** Validated on-disk compilation cache for generated model wrappers. */
public final class ModelCompilationCache {
	private ModelCompilationCache() {}

	public static LoadedGeneratedModel compile(String script, Path cacheDirectory)
			throws IOException {
		if (script == null || cacheDirectory == null)
			throw new IllegalArgumentException("script and cache directory are required");
		// Validate before writing or invoking javac.
		ModelScript.validateSyntax(script);
		String hash = ModelSourceGenerator.hash(script);
		String packageName = "jdistlib.generated";
		String simpleName = "Model_" + hash.substring(0, 24);
		String className = packageName + "." + simpleName;
		Path root = cacheDirectory.toAbsolutePath().normalize();
		Files.createDirectories(root);
		Path packageDirectory = root.resolve("jdistlib").resolve("generated").normalize();
		if (!packageDirectory.startsWith(root)) throw new SecurityException("generated path escaped cache root");
		Files.createDirectories(packageDirectory);
		Path sourceFile = packageDirectory.resolve(simpleName + ".java");
		Path classFile = packageDirectory.resolve(simpleName + ".class");
		if (!Files.isRegularFile(classFile)) {
			Files.write(sourceFile, ModelSourceGenerator.generate(className, script)
					.getBytes(StandardCharsets.UTF_8));
			compileJava(root, sourceFile);
		}
		try {
			URLClassLoader loader = new URLClassLoader(new URL[] {root.toUri().toURL()},
					ModelCompilationCache.class.getClassLoader());
			Class<?> generated = Class.forName(className, true, loader);
			Object instance = generated.getDeclaredConstructor().newInstance();
			if (!(instance instanceof GeneratedModelFactory)) {
				loader.close(); throw new IOException("generated class does not implement its contract");
			}
			GeneratedModelFactory factory = (GeneratedModelFactory) instance;
			if (!hash.equals(factory.sourceHash())) {
				loader.close(); throw new IOException("generated-model cache hash mismatch");
			}
			return new LoadedGeneratedModel(factory, loader);
		} catch (ReflectiveOperationException exception) {
			throw new IOException("could not load generated model", exception);
		}
	}

	private static void compileJava(Path output, Path source) throws IOException {
		JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
		if (compiler == null) throw new IOException("a JDK compiler is required for generated models");
		DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<JavaFileObject>();
		try (StandardJavaFileManager manager = compiler.getStandardFileManager(
				diagnostics, null, StandardCharsets.UTF_8)) {
			Iterable<? extends JavaFileObject> units = manager.getJavaFileObjects(source.toFile());
			List<String> options = Arrays.asList("-d", output.toString(), "-classpath",
					System.getProperty("java.class.path"), "-encoding", "UTF-8");
			Boolean success = compiler.getTask(null, manager, diagnostics, options,
					null, units).call();
			if (!Boolean.TRUE.equals(success)) {
				List<ScriptDiagnostic> messages = new ArrayList<ScriptDiagnostic>();
				for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics())
					messages.add(new ScriptDiagnostic((int) diagnostic.getLineNumber(),
							(int) diagnostic.getColumnNumber(), diagnostic.getMessage(null)));
				throw new ModelScriptException(messages);
			}
		}
	}
}
