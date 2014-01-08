package jdistlib.util;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;

import java.util.Arrays;

import jdistlib.math.MathFunctions;

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


	public static final double[] pows(double x, double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] pows(double x, int[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] mins(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = -e[i];
		return v;
	}

	public static final double[] mins(double[] a, double[] b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]-b[i];
		return v;
	}

	public static final double[] times(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a*b[i];
		return v;
	}

	public static final double[] divs(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b;
		return v;
	}

	public static final double[] divs(double[] a, double[] b) {
		double[] v = new double[a.length];
		if (a.length != b.length)
			throw new RuntimeException();
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b[i];
		return v;
	}

	public static final double[] comps(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = 1-e[i];
		return v;
	}

	public static final double[] abss(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = abs(e[i]);
		return v;
	}

	public static final double[] exps(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = exp(e[i]);
		return v;
	}

	public static final double[] logs(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log(e[i]);
		return v;
	}

	public static final double[] log1pComps(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log1p(-e[i]);
		return v;
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

	public static final double[] diff(double[] e, int lag, int order) {
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, e.length);
		v = diff_impl(v, v.length, lag, order);
		e = new double[e.length - lag * order];
		System.arraycopy(v, 0, e, 0, e.length);
		return e;
	}

	private static final double[] diff_impl(double[] e, int elen, int lag, int order) {
		for (int i = lag; i < elen; i++)
			e[i - lag] = e[i] - e[i - lag];
		if (order > 1)
			diff_impl(e, elen - lag, lag, order - 1);
		return e;
	}

	public static final boolean allFinite(double[] e) {
		for (double _e : e)
			if (MathFunctions.isInfinite(_e))
				return false;
		return true;
	}

	public static final boolean allLt(double[] e, double v) {
		for (double _e : e)
			if (_e >= v)
				return false;
		return true;
	}
}
