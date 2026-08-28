/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

/** Immutable names and grouping labels for pointwise likelihood contributions. */
public final class ObservationMetadata {
	private final String[] names;
	private final String[] groups;

	public ObservationMetadata(String[] names, String[] groups) {
		if (names == null || groups == null || names.length != groups.length)
			throw new IllegalArgumentException("matching observation names and groups are required");
		this.names = names.clone(); this.groups = groups.clone();
		for (int i = 0; i < names.length; i++) {
			if (!valid(names[i]) || !valid(groups[i]))
				throw new IllegalArgumentException("observation names and groups must not be blank");
		}
	}

	public static ObservationMetadata ungrouped(String... names) {
		if (names == null) throw new IllegalArgumentException("observation names are required");
		return new ObservationMetadata(names, names);
	}

	public int size() { return names.length; }
	public String name(int index) { return names[index]; }
	public String group(int index) { return groups[index]; }
	public String[] names() { return names.clone(); }
	public String[] groups() { return groups.clone(); }

	static ObservationMetadata concatenate(ObservationMetadata[] values) {
		int count = 0;
		for (ObservationMetadata value : values) {
			if (value == null) throw new IllegalArgumentException("observation metadata must not be null");
			count += value.size();
		}
		String[] names = new String[count], groups = new String[count]; int offset = 0;
		for (ObservationMetadata value : values) {
			System.arraycopy(value.names, 0, names, offset, value.size());
			System.arraycopy(value.groups, 0, groups, offset, value.size());
			offset += value.size();
		}
		return new ObservationMetadata(names, groups);
	}

	private static boolean valid(String value) { return value != null && !value.trim().isEmpty(); }

	@Override public boolean equals(Object other) {
		if (!(other instanceof ObservationMetadata)) return false;
		ObservationMetadata value = (ObservationMetadata) other;
		return Arrays.equals(names, value.names) && Arrays.equals(groups, value.groups);
	}
	@Override public int hashCode() { return 31 * Arrays.hashCode(names) + Arrays.hashCode(groups); }
}
