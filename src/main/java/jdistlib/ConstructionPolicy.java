/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Determines which advisory findings prevent analyzed construction. */
public enum ConstructionPolicy {
	/** Any warning or error prevents construction. */
	STRICT,
	/** Errors prevent construction; warnings are retained in the report. */
	WARNING,
	/** Analysis never prevents an attempt, though hard numerical failures still do. */
	PERMISSIVE
}
