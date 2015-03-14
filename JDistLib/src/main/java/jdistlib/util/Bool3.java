package jdistlib.util;

/**
 * Three-state boolean: TRUE, FALSE, NA
 * @author Roby Joehanes
 *
 */
public enum Bool3 {
	TRUE, FALSE, NA;
	public boolean v() {
		if (this == NA)
			throw new RuntimeException();
		if (this == TRUE)
			return true;
		return false;
	}
}
