/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.Map;

/** Contract implemented by ahead-of-time generated script wrappers. */
public interface GeneratedModelFactory {
	CompiledModelScript compile(Map<String, double[]> data);
	String sourceHash();
}
