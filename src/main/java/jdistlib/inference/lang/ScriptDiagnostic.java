/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

/** Source-located modeling-language diagnostic. */
public final class ScriptDiagnostic implements java.io.Serializable {
	private static final long serialVersionUID = 1L;
	private final int line;
	private final int column;
	private final String message;
	public ScriptDiagnostic(int line, int column, String message) {
		this.line = line; this.column = column; this.message = message;
	}
	public int line() { return line; }
	public int column() { return column; }
	public String message() { return message; }
	@Override public String toString() { return line + ":" + column + ": " + message; }
}
