package jdistlib.util;

import java.util.Arrays;
import java.util.LinkedHashSet;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/**
 * Utility functions to mimic R
 * @author Roby Joehanes
 *
 */
public class Utilities {
	public enum RankTies { AVERAGE, MAX, MIN }

	public static final int[] colon(int from, int to) {
		int n = Math.abs(to - from) + 1;
		int[] d = new int[n];
		int inc = to > from ? 1 : -1;
		for (int i = 0 ; i < n; i++)
			d[i] = from + i*inc;
		return d;
	}

	public static final double[] colon(double from, double to) {
		int n = (int) (Math.abs(to - from) + 1);
		double[] d = new double[n];
		int inc = to > from ? 1 : -1;
		for (int i = 0 ; i < n; i++)
			d[i] = from + i*inc;
		return d;
	}

	public static final int[] seq(int from, int to, int by) {
		int n = (to - from) / by + 1;
		int[] d = new int[n];
		for (int i = 0 ; i < n; i++)
			d[i] = from + i * by;
		return d;
	}

	public static final double[] seq(double from, double to, double by) {
		int n = (int) Math.ceil((to - from + 1e-15) / by);
		double[] d = new double[n];
		for (int i = 0 ; i < n; i++)
			d[i] = from + i * by;
		return d;
	}

	public static final int[] c(int[]... x) {
		int n = 0;
		for (int i = 0; i < x.length; i++)
			n += x[i].length;
		int[] v = new int[n];
		int w = 0;
		for (int i = 0; i < x.length; i++) {
			System.arraycopy(x[i], 0, v, w, x[i].length);
			w += x[i].length;
		}
		return v;
	}

	public static final int[] c(int... x) {
		return x;
	}

	public static final double[] c(double[]... x) {
		int n = 0;
		for (int i = 0; i < x.length; i++)
			n += x[i].length;
		double[] v = new double[n];
		int w = 0;
		for (int i = 0; i < x.length; i++) {
			System.arraycopy(x[i], 0, v, w, x[i].length);
			w += x[i].length;
		}
		return v;
	}

	public static final double[] c(double... x) {
		return x;
	}

	public static final double[] rep(double v, int n) {
		double[] r = new double[n];
		Arrays.fill(r, v);
		return r;
	}

	public static final double[] rep(double[] v, int n) {
		double[] r = new double[n * v.length];
		for (int i = 0; i < n; i++)
			System.arraycopy(v, 0, r, i * n, v.length);
		return r;
	}

	public static final double[] rep_each(double[] v, int n) {
		int vn = v.length;
		double[] r = new double[vn];
		for (int i = 0; i < vn; i++) {
			double vi = v[i];
			for (int j = 0; j < n; j++)
				r[i*n + j] = vi;
		}
		return r;
	}

	public static final double[] rec(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = 1.0/e[i];
		return v;
	}

	public static final double[] rev(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[e.length - i - 1] = e[i];
		return v;
	}

	public static final void print(double... val) {
		print(" %g", val);
	}

	public static final void print(String format, double... val) {
		int n = val.length;
		for (int i = 0; i < n; i++) {
			System.out.print(String.format(format, val[i]));
			if ((i + 1) % 6 == 0) System.out.println();
		}
		System.out.println();
	}

	/**
	 * Sort data using radix sort
	 * @param data
	 */
	public static final void sort(double[] data) {
		// N=5*10^8: Radix = 9919, Quick = 16971 (ms).
		int
			n = data.length,
			numBins = (int) Math.max(4,Math.min(10, Math.round(Math.log(n) / (2 * Math.log(2)))));
		long[]
			a = new long[n],
			b = new long[n];
		for (int i = 0; i < n; i++)
			a[i] = Double.doubleToLongBits(data[i]);
		for (long mask = ~(-1L << numBins), rshift = 0L; mask != 0; mask <<= numBins, rshift += numBins) {
			int[] cntarray = new int[1 << numBins];
			for (int p = 0; p < n; ++p)
				++cntarray[(int) ((a[p] & mask) >>> rshift)];
			for (int i = 1; i < cntarray.length; ++i)
				cntarray[i] += cntarray[i-1];
			for (int p = n-1; p >= 0; --p) {
				int key = (int) ((a[p] & mask) >>> rshift);
				--cntarray[key];
				b[cntarray[key]] = a[p];
			}
			long[] temp = b; b = a; a = temp;
		}

		int numNegs = 0;
		// Negatives are always placed at the end of the array in the reverse order.
		// So, scan to find the point
		if (a[n - 1] < 0)
			for (int i = n - 1; a[i] < 0; i--)
				data[numNegs++] = Double.longBitsToDouble(a[i]);
		if (numNegs > 0)
			for (int i = numNegs; i < n; i++)
				data[i] = Double.longBitsToDouble(a[i - numNegs]);
		else
			for (int i = 0; i < n; i++)
				data[i] = Double.longBitsToDouble(a[i]);
	}

	/**
	 * Sort using radix sort
	 * @param data
	 * @param idx an index of data, of the same length, useful to sort an entire table
	 */
	public static final void sort(double[] data, int[] idx) {
		// N=5*10^8: Radix = 9919, Quick = 16971 (ms).
		int
			n = data.length,
			numBins = (int) Math.max(4,Math.min(10, Math.round(Math.log(n) / (2 * Math.log(2)))));
		long[]
			a = new long[n],
			b = new long[n];
		int[]
			idx_a = new int[n],
			idx_b = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = Double.doubleToLongBits(data[i]);
			idx_a[i] = idx[i];
		}
		for (long mask = ~(-1L << numBins), rshift = 0L; mask != 0; mask <<= numBins, rshift += numBins) {
			int[] cntarray = new int[1 << numBins];
			for (int p = 0; p < n; ++p)
				++cntarray[(int) ((a[p] & mask) >>> rshift)];
			for (int i = 1; i < cntarray.length; ++i)
				cntarray[i] += cntarray[i-1];
			for (int p = n-1; p >= 0; --p) {
				int key = (int) ((a[p] & mask) >>> rshift);
				--cntarray[key];
				b[cntarray[key]] = a[p];
				idx_b[cntarray[key]] = idx_a[p];
			}
			long[] temp = b; b = a; a = temp;
			int[] temp_idx = idx_b; idx_b = idx_a; idx_a = temp_idx;
		}

		int numNegs = 0;
		// Negatives are always placed at the end of the array in the reverse order.
		// So, scan to find the point
		if (a[n - 1] < 0)
			for (int i = n - 1; a[i] < 0; i--) {
				data[numNegs] = Double.longBitsToDouble(a[i]);
				idx[numNegs++] = idx_a[i];
			}
		if (numNegs > 0)
			for (int i = numNegs; i < n; i++) {
				data[i] = Double.longBitsToDouble(a[i - numNegs]);
				idx[i] = idx_a[i - numNegs];
			}
		else
			for (int i = 0; i < n; i++) {
				data[i] = Double.longBitsToDouble(a[i]);
				idx[i] = idx_a[i];
			}
	}

	/**
	 * Radix sort
	 * @param a
	 */
	public static final void sort(int[] a) {
		int
			n = a.length,
			// Roby's modification: Optimum number of bins
			numBins = (int) Math.max(4,Math.min(10, Math.round(Math.log(n) / (2 * Math.log(2))))),
			b[] = new int[n],
			b_orig[] = b;
		for (int mask = ~(-1 << numBins), rshift = 0; mask != 0; mask <<= numBins, rshift += numBins) {
			int[] cntarray = new int[1 << numBins];
			for (int p = 0; p < n; ++p)
				++cntarray[(a[p] & mask) >>> rshift]; // RJ's fix to handle negatives
			for (int i = 1; i < cntarray.length; ++i)
				cntarray[i] += cntarray[i-1];
			for (int p = n-1; p >= 0; --p) {
				int key = (a[p] & mask) >>> rshift; // RJ's fix to handle negatives
				--cntarray[key];
				b[cntarray[key]] = a[p];
			}
			int[] temp = b; b = a; a = temp;
		}
		if (a == b_orig)
			System.arraycopy(a, 0, b, 0, n);
		int numNegs = 0;
		// Negatives are always placed at the end of the array in the correct order.
		// So, scan to find the point
		for (int i = n - 1; a[i] < 0; i--, numNegs++) ;
		if (numNegs > 0) {
			System.arraycopy(a, n - numNegs, b, 0, numNegs);
			System.arraycopy(a, 0, a, numNegs, n - numNegs);
			System.arraycopy(b, 0, a, 0, numNegs);
		}
	}

	/**
	 * Converts the integer array e into a double array
	 * @param e
	 * @return a double array
	 */
	public static final double[] to_double(int[] e) {
		int n = e.length;
		double[] r = new double[n];
		for (int i = 0; i < n; i++)
			r[i] = e[i];
		return r;
	}

	/**
	 * Returns the order of the elements of array e
	 * @param e
	 */
	public static final int[] order(double[] e) {
		int n = e.length;
		int[] order = new int[n];
		double[] v = new double[n];
		System.arraycopy(e, 0, v, 0, n);
		for (int i = 0; i < n; i++)
			order[i] = i;
		sort(v, order);
		return order;
	}

	/**
	 * Returns the ranks of the elements of array e, resolve ties by averaging
	 * @param e
	 */
	public static final double[] rank(double[] e) {
		return rank(e, RankTies.AVERAGE);
	}

	/**
	 * Returns the ranks of the elements of array e
	 * @param e
	 * @param ties Ties resolution: Average, Min, or Max
	 */
	public static final double[] rank(double[] e, RankTies ties) {
		int n = e.length;
		int[] order = new int[n];
		double[] rank = new double[n];
		double[] v = new double[n];
		System.arraycopy(e, 0, v, 0, n);
		for (int i = 0; i < n; i++)
			order[i] = i;
		sort(v, order);
		int j;
		for (int i = 0; i < n; i = j+1) {
			j = i;
			while ((j < n - 1) && (v[j] == v[j+1])) j++;
			switch (ties) {
				case AVERAGE:
					double avg = (i + j) / 2.;
					for (int k = i; k <= j; k++)
						rank[order[k]] = avg;
					break;
				case MAX:
					for (int k = i; k <= j; k++)
						rank[order[k]] = j;
					break;
				case MIN:
					for (int k = i; k <= j; k++)
						rank[order[k]] = i;
					break;
			}
		}
		return rank;
	}

	/**
	 * Permute the array e
	 * @param e
	 */
	public static final void permute(double[] e) {
		permute(e, new MersenneTwister());
	}

	/**
	 * Permute the array e
	 * @param e
	 * @param random
	 */
	public static final void permute(double[] e, RandomEngine random) {
		int n = e.length;
		for (int i = 0; i < n; i++) {
			int
				i1 = random.nextInt(n),
				i2 = random.nextInt(n);
			double t = e[i1]; e[i1] = e[i2]; e[i2] = t;
		}
	}

	public static final double[] unique(double[] e) {
		LinkedHashSet<Double> set = new LinkedHashSet<Double>(e.length);
		for (double _e : e)
			set.add(_e);
		int n = e.length, i = 0;
		double[] r = new double[n];
		for (double _e: set)
			r[i++] = _e;
		return r;
	}

	public static final <S> S[] unique(S[] e) {
		LinkedHashSet<S> set = new LinkedHashSet<S>(e.length);
		for (S _e : e)
			set.add(_e);
		return (S[]) set.toArray();
	}

	public static final boolean[] is_duplicated(double[] e) {
		LinkedHashSet<Double> set = new LinkedHashSet<Double>(e.length);
		int n = e.length, i = 0;
		boolean[] r = new boolean[n];
		for (double _e : e) {
			r[i++] = set.contains(_e);
			set.add(_e);
		}
		return r;
	}

	public static final <S> boolean[] is_duplicated(S[] e) {
		LinkedHashSet<S> set = new LinkedHashSet<S>(e.length);
		int n = e.length, i = 0;
		boolean[] r = new boolean[n];
		for (S _e : e) {
			r[i++] = set.contains(_e);
			set.add(_e);
		}
		return r;
	}

	public static final int[] which(boolean[] e) {
		int n = e.length, n_true = 0, j = 0;
		for (boolean _e : e)
			if (_e) n_true++;
		int[] idx = new int[n_true];
		for (int i = 0; i < n; i++)
			if (e[i])
				idx[j++] = i;
		return idx;
	}
}
