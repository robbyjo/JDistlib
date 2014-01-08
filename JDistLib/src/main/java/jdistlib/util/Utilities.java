package jdistlib.util;

import java.util.Arrays;

/**
 * Utility functions to mimic R
 * @author Roby Joehanes
 *
 */
public class Utilities {
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
}
