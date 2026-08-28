/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.HashSet;
import java.util.Set;

/** Named parameter schema for one model in a trans-dimensional target. */
public final class ReversibleJumpModelSpace {
	private final long modelId; private final String name; private final String[] parameterNames;
	public ReversibleJumpModelSpace(long modelId, String name, String... parameterNames) {
		if (modelId < 0L || name == null || name.trim().isEmpty() || parameterNames == null)
			throw new IllegalArgumentException("model id, name, and parameter names required");
		this.modelId = modelId; this.name = name; this.parameterNames = parameterNames.clone();
		Set<String> unique = new HashSet<String>();
		for (String parameter : this.parameterNames)
			if (parameter == null || parameter.trim().isEmpty() || !unique.add(parameter))
				throw new IllegalArgumentException("parameter names must be nonblank and unique");
	}
	public long modelId() { return modelId; }
	public String name() { return name; }
	public int dimension() { return parameterNames.length; }
	public String parameterName(int index) { return parameterNames[index]; }
	public String[] parameterNames() { return parameterNames.clone(); }
}
