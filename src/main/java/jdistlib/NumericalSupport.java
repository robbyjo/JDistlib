/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Immutable union of continuous intervals, optional atoms, and singularities. */
public final class NumericalSupport {
	/** One nonempty continuous interval. Endpoint inclusion is immaterial to density. */
	public static final class Interval {
		private final double lower;
		private final double upper;
		private Interval(double lower, double upper) {
			this.lower = lower;
			this.upper = upper;
		}
		public double getLower() { return lower; }
		public double getUpper() { return upper; }
		public boolean contains(double x) { return x >= lower && x <= upper; }
	}

	private final Interval[] intervals;
	private final double[] atoms;
	private final double[] singularities;
	private final double lowerBound;
	private final double upperBound;

	private NumericalSupport(Interval[] intervals, double[] atoms,
			double[] singularities) {
		this.intervals = intervals;
		this.atoms = atoms;
		this.singularities = singularities;
		double lower = Double.POSITIVE_INFINITY;
		double upper = Double.NEGATIVE_INFINITY;
		for (Interval interval : intervals) {
			lower = Math.min(lower, interval.lower);
			upper = Math.max(upper, interval.upper);
		}
		for (double atom : atoms) {
			lower = Math.min(lower, atom);
			upper = Math.max(upper, atom);
		}
		lowerBound = lower;
		upperBound = upper;
	}

	public static Builder builder() { return new Builder(); }
	public static NumericalSupport interval(double lower, double upper) {
		return builder().interval(lower, upper).build();
	}

	public Interval[] getIntervals() { return intervals.clone(); }
	public double[] getAtoms() { return atoms.clone(); }
	public double[] getSingularities() { return singularities.clone(); }
	public double getLowerBound() { return lowerBound; }
	public double getUpperBound() { return upperBound; }
	public boolean hasAtoms() { return atoms.length != 0; }

	public boolean containsContinuous(double x) {
		for (Interval interval : intervals) if (interval.contains(x)) return true;
		return false;
	}

	/** Builder supporting interval unions followed by hole subtraction. */
	public static final class Builder {
		private final List<Interval> intervals = new ArrayList<Interval>();
		private final List<Interval> holes = new ArrayList<Interval>();
		private final List<Double> atoms = new ArrayList<Double>();
		private final List<Double> singularities = new ArrayList<Double>();

		private Builder() {}

		public Builder interval(double lower, double upper) {
			intervals.add(validInterval(lower, upper, "interval"));
			return this;
		}

		/** Removes an open interval from every declared continuous interval. */
		public Builder hole(double lower, double upper) {
			holes.add(validInterval(lower, upper, "hole"));
			return this;
		}

		public Builder atom(double location) {
			if (!Double.isFinite(location)) {
				throw new IllegalArgumentException("atom locations must be finite");
			}
			atoms.add(location);
			return this;
		}

		/** Declares a finite split point for integration. */
		public Builder singularity(double location) {
			if (!Double.isFinite(location)) {
				throw new IllegalArgumentException("singularities must be finite");
			}
			singularities.add(location);
			return this;
		}

		public NumericalSupport build() {
			List<Interval> merged = merge(intervals);
			for (Interval hole : holes) merged = subtract(merged, hole);
			double[] atomArray = sortedUnique(atoms, "atoms");
			if (merged.isEmpty() && atomArray.length == 0) {
				throw new IllegalArgumentException("support must contain an interval or atom");
			}
			double[] splitArray = sortedUnique(singularities, "singularities");
			List<Double> retained = new ArrayList<Double>();
			for (double split : splitArray) {
				for (Interval interval : merged) {
					if (split > interval.lower && split < interval.upper) {
						retained.add(split);
						break;
					}
				}
			}
			double[] retainedArray = new double[retained.size()];
			for (int i = 0; i < retainedArray.length; i++) retainedArray[i] = retained.get(i);
			return new NumericalSupport(merged.toArray(new Interval[merged.size()]),
					atomArray, retainedArray);
		}

		private static Interval validInterval(double lower, double upper,
				String description) {
			if (Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper)
					|| lower == Double.POSITIVE_INFINITY
					|| upper == Double.NEGATIVE_INFINITY) {
				throw new IllegalArgumentException(description + " bounds are invalid");
			}
			return new Interval(lower, upper);
		}

		private static List<Interval> merge(List<Interval> source) {
			List<Interval> sorted = new ArrayList<Interval>(source);
			Collections.sort(sorted, Comparator.comparingDouble(interval -> interval.lower));
			List<Interval> result = new ArrayList<Interval>();
			for (Interval interval : sorted) {
				if (result.isEmpty()
						|| interval.lower > result.get(result.size() - 1).upper) {
					result.add(interval);
				} else {
					Interval previous = result.remove(result.size() - 1);
					result.add(new Interval(previous.lower,
							Math.max(previous.upper, interval.upper)));
				}
			}
			return result;
		}

		private static List<Interval> subtract(List<Interval> source, Interval hole) {
			List<Interval> result = new ArrayList<Interval>();
			for (Interval interval : source) {
				if (hole.upper <= interval.lower || hole.lower >= interval.upper) {
					result.add(interval);
					continue;
				}
				if (hole.lower > interval.lower) {
					result.add(new Interval(interval.lower,
							Math.min(hole.lower, interval.upper)));
				}
				if (hole.upper < interval.upper) {
					result.add(new Interval(Math.max(hole.upper, interval.lower),
							interval.upper));
				}
			}
			return result;
		}

		private static double[] sortedUnique(List<Double> values, String name) {
			double[] result = new double[values.size()];
			for (int i = 0; i < result.length; i++) result[i] = values.get(i);
			Arrays.sort(result);
			for (int i = 1; i < result.length; i++) {
				if (result[i] == result[i - 1]) {
					throw new IllegalArgumentException(name + " must be unique");
				}
			}
			return result;
		}
	}
}
