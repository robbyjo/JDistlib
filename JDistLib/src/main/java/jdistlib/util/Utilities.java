package jdistlib.util;

/**
 * Utility functions to mimic R
 * @author Roby Joehanes
 *
 */
public class Utilities {
	public static final int[] colon(int from, int to) {
		int n = (to - from) + 1;
		int[] d = new int[n];
		for (int i = 0 ; i < n; i++)
			d[i] = from + i;
		return d;
	}

	public static final double[] colon(double from, double to) {
		int n = (int) ((to - from) + 1);
		double[] d = new double[n];
		for (int i = 0 ; i < n; i++)
			d[i] = from + i;
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
}
