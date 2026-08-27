/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Parse, validation, or compilation failure with source diagnostics. */
public final class ModelScriptException extends IllegalArgumentException {
	private static final long serialVersionUID = 1L;
	private final ArrayList<ScriptDiagnostic> diagnostics;
	public ModelScriptException(List<ScriptDiagnostic> diagnostics) {
		super(diagnostics == null || diagnostics.isEmpty() ? "model script failed"
				: diagnostics.get(0).toString());
		this.diagnostics = new ArrayList<ScriptDiagnostic>(diagnostics);
	}
	public List<ScriptDiagnostic> diagnostics() {
		return Collections.unmodifiableList(diagnostics);
	}
}
