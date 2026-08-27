/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.io.IOException;
import java.net.URLClassLoader;

/** Closeable generated-model factory and its isolated class loader. */
public final class LoadedGeneratedModel implements AutoCloseable {
	private final GeneratedModelFactory factory;
	private final URLClassLoader loader;
	LoadedGeneratedModel(GeneratedModelFactory factory, URLClassLoader loader) {
		this.factory = factory; this.loader = loader;
	}
	public GeneratedModelFactory factory() { return factory; }
	@Override public void close() throws IOException { loader.close(); }
}
