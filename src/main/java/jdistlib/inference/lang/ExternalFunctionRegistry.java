/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Immutable name-to-Java registry for forward-declared Stan functions. */
public final class ExternalFunctionRegistry {
	private final Map<String, StanExternalFunction> functions;
	private ExternalFunctionRegistry(Map<String, StanExternalFunction> functions) {
		this.functions = Collections.unmodifiableMap(new LinkedHashMap<String, StanExternalFunction>(functions));
	}
	StanExternalFunction function(String name) { return functions.get(name); }
	boolean contains(String name) { return functions.containsKey(name); }
	public static Builder builder() { return new Builder(); }
	public static ExternalFunctionRegistry empty() { return new Builder().build(); }

	public static final class Builder {
		private final Map<String, StanExternalFunction> functions = new LinkedHashMap<String, StanExternalFunction>();
		public Builder bind(String name, StanExternalFunction function) {
			if (name == null || !name.matches("[A-Za-z_][A-Za-z0-9_]*") || function == null)
				throw new IllegalArgumentException("valid external function name and callback required");
			if (functions.put(name, function) != null)
				throw new IllegalArgumentException("duplicate external function binding: "+name);
			return this;
		}
		public ExternalFunctionRegistry build() { return new ExternalFunctionRegistry(functions); }
	}
}
