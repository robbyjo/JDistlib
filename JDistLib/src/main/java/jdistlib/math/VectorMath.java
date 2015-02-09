/*  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package jdistlib.math;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jdistlib.util.Utilities.sort;

/**
 * 
 * @author Roby Joehanes
 *
 */
public class VectorMath {
	public static final double[] vpow(double x, double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] vpow(double[] x, double e) {
		double[] v = new double[x.length];
		for (int i = 0; i < x.length; i++)
			v[i] = pow(x[i], e);
		return v;
	}

	public static final double[] vpow(double[] x, double[] e) {
		if (x.length != e.length)
			throw new RuntimeException();
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x[i], e[i]);
		return v;
	}

	public static final double[] vpow(double x, int[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] vplus(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]+b[i];
		return v;
	}

	public static final double[] vplus(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a+b[i];
		return v;
	}

	public static final double[] vplus(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]+b;
		return v;
	}

	public static final double[] vmin(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = -e[i];
		return v;
	}

	public static final double[] vmin(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]-b[i];
		return v;
	}

	public static final double[] vmin(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]-b;
		return v;
	}

	public static final double[] vmin(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a-b[i];
		return v;
	}

	public static final double[] vtimes(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a*b[i];
		return v;
	}

	public static final double[] vtimes(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]*b;
		return v;
	}

	public static final double[] vtimes(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]*b[i];
		return v;
	}

	public static final double[] vsq(double[] a) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]*a[i];
		return v;
	}

	public static final double[] vdiv(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b;
		return v;
	}

	public static final double[] vdiv(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a/b[i];
		return v;
	}

	public static final double[] vdiv(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b[i];
		return v;
	}

	public static final double[] vcomp(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = 1-e[i];
		return v;
	}

	public static final double[] vabs(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = abs(e[i]);
		return v;
	}

	public static final double[] vexp(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = exp(e[i]);
		return v;
	}

	public static final double[] vlog(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log(e[i]);
		return v;
	}

	public static final double[] vlog1pComps(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log1p(-e[i]);
		return v;
	}

	public static final double[] vsignif(double[] e, int digits) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = MathFunctions.signif(e[i], digits);
		return v;
	}

	public static final double[] diff(double[] e, int lag, int order) {
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, e.length);
		int vlen = v.length;
		for (int i = 0; i < order; i++, vlen -= lag)
			for (int j = lag; j < vlen; j++)
				v[j - lag] = v[j] - v[j - lag];
		e = new double[e.length - lag * order];
		System.arraycopy(v, 0, e, 0, e.length);
		return e;
	}

	public static final double[] diff(double[] e, int lag) {
		return diff(e, lag, 1);
	}

	public static final double[] diff(double[] e) {
		return diff(e, 1, 1);
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

	public static final boolean allGt(double[] e, double v) {
		for (double _e : e)
			if (_e <= v)
				return false;
		return true;
	}

	public static final boolean allEq(double[] e, double v) {
		for (double _e : e)
			if (_e != v)
				return false;
		return true;
	}

	public static final boolean allEqual(double[] e, double[] v) {
		if (e.length != v.length)
			throw new RuntimeException();
		int n = e.length;
		for (int i = 0; i < n; i++)
			if (e[i] != v[i])
				return false;
		return true;
	}

	/**
	 * Find quantile given a sorted data of array (Definition 7)
	 * @param sortedData This data is assumed to be presorted! Use quantile0 if you want to use unsorted data!
	 * @param quantile must be 0 <= quantile <= 1
	 * @return quantile value
	 */
	public static final double quantile(double[] sortedData, double quantile) {
		double index = (sortedData.length - 1) * quantile;
		int
			lo = (int) Math.floor(index),
			hi = (int) Math.ceil(index);
		if (lo < 0)
			return 0;
		double
			h = index - lo,
			lowerQ = sortedData[lo],
			result = h == 0 ? lowerQ : (1 - h) * lowerQ + h * sortedData[hi];
		return result;
	}

	/**
	 * Find quantile given a sorted data of array (Definition 7)
	 * @param sortedData This data is assumed to be presorted!
	 * @param quantile
	 * @return quantile values
	 */
	public static final double[] quantile(double[] sortedData, double[] quantile) {
		double[] v = new double[quantile.length];
		for (int i = 0; i < quantile.length; i++)
			v[i] = quantile(sortedData, quantile[i]);
		return v;
	}

	/**
	 * Find quantile in an array (Definition 7). Data is assumed to be unsorted
	 * @param e
	 * @param quantile must be 0 <= quantile <= 1
	 * @return quantile value
	 */
	public static final double quantile0(double[] e, double quantile) {
		int n = e.length;
		double[] r = new double[n];
		System.arraycopy(e, 0, r, 0, n);
		sort(r);
		return quantile(r, quantile);
	}

	/**
	 * Find quantile in an array (Definition 7). Data is assumed to be unsorted
	 * @param e
	 * @param quantile must be 0 <= quantile <= 1
	 * @return quantile values
	 */
	public static final double[] quantile0(double[] e, double[] quantile) {
		int n = e.length;
		double[] r = new double[n];
		System.arraycopy(e, 0, r, 0, n);
		sort(r);
		double[] v = new double[quantile.length];
		for (int i = 0; i < quantile.length; i++)
			v[i] = quantile(r, quantile[i]);
		return v;
	}

	public static final double mean(double[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += (e[i] / n); // guard against overflow
		return sum;
	}

	/**
	 * Get the median
	 * @param e does not need to be sorted
	 * @return median value
	 */
	public static final double median(double[] e) {
		int n = e.length;
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, n);
		sort(v);
		return quantile(v, 0.5);
	}

	public static final double sd(double[] e) {
		double sum = 0, sumsq = 0;
		int n = e.length, nm1 = n-1;
		for (int i = 0; i < n; i++) {
			double v = e[i]; // guard against overflow
			sum += v / n;
			sumsq += v * v / nm1;
		}
		return sqrt(sumsq - (sum / nm1) * sum * n);
	}

	public static final double var(double[] e) {
		double sum = 0, sumsq = 0;
		int n = e.length, nm1 = n-1;
		for (int i = 0; i < n; i++) {
			double v = e[i]; // guard against overflow
			sum += v / n;
			sumsq += v * v / nm1;
		}
		return sumsq - (sum / nm1) * sum * n;
	}

	public static final double sum(double[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += e[i];
		return sum;
	}

	public static final double sum_kahan(double[] e) {
		double sum = e[0], c = 0;
		int n = e.length;
		for (int i = 1; i < n; i++) {
			double y = e[i] - c,
				t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
		return sum - c;
	}

	public static final double sum(Map<String, Integer> e) {
		double sum = 0;
		for (int v: e.values())
			sum += v;
		return sum;
	}

	/**
	 * Standardize the value in x (i.e., (x - mean(x)) / sd(x))
	 * @param x
	 * @return standardized values
	 */
	public static final double[] standardize(double[] x) {
		double sum = 0, sumsq = 0;
		int n = x.length, nm1 = n-1;
		double[] new_x = new double[n];
		for (int i = 0; i < n; i++) {
			double v = x[i];
			sum += v / n; // guard against overflow
			sumsq += v * v / nm1;
		}
		sumsq = sqrt(sumsq - (sum / nm1) * sum * n);
		for (int i = 0; i < n; i++) {
			new_x[i] = (x[i] - sum) / sumsq;
		}
		return new_x;
	}

	public static final Map<String, Integer> table(double[] e) {
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (double _e : e) {
			String estr = String.valueOf(_e);
			Integer i = map.get(estr);
			map.put(estr, 1 + (i == null ? 0 : i.intValue()));
		}
		return map;
	}

	public static final <T> Map<String, Integer> table(T[] e) {
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (T _e : e) {
			String estr = String.valueOf(_e);
			Integer i = map.get(estr);
			map.put(estr, 1 + (i == null ? 0 : i.intValue()));
		}
		return map;
	}

	public static final double[] as_numeric(Collection<String> ll) {
		int n = ll.size(), i = 0;
		double[] v = new double[n];
		for (String str: ll) {
			v[i] = Double.valueOf(str);
			i++;
		}
		return v;
	}

	public static final double[] cumsum(double[] e) {
		int n = e.length;
		double[] r = new double[n];
		r[0] = e[0];
		for (int i = 1; i < n; i++)
			r[i] = r[i-1] + e[i];
		return r;
	}

	public static final int[] cumsum(int[] e) {
		int n = e.length;
		int[] r = new int[n];
		r[0] = e[0];
		for (int i = 1; i < n; i++)
			r[i] = r[i-1] + e[i];
		return r;
	}

	public static final double max(double[] e) {
		int n = e.length;
		double mx = e[0];
		for (int i = 1; i < n; i++)
			if (e[i] > mx) mx = e[i];
		return mx;
	}

	public static final double min(double[] e) {
		int n = e.length;
		double mn = e[0];
		for (int i = 1; i < n; i++)
			if (e[i] < mn) mn = e[i];
		return mn;
	}

	public static final int which_max(double[] e) {
		int n = e.length, which = 0;
		double mx = e[0];
		for (int i = 1; i < n; i++)
			if (e[i] > mx) { mx = e[i]; which = i; };
		return which;
	}

	public static final int which_min(double[] e) {
		int n = e.length, which = 0;
		double mn = e[0];
		for (int i = 1; i < n; i++)
			if (e[i] < mn) { mn = e[i]; which = i; };
		return which;
	}

	public static final double[] pmax(double[] a, double[] b) {
		int n = a.length;
		if (n != b.length)
			throw new RuntimeException();
		double[] mx = new double[n];
		for (int i = 0; i < n; i++)
			mx[i] = a[i] > b[i] ? a[i] : b[i];
		return mx;
	}

	public static final double[] pmin(double[] a, double[] b) {
		int n = a.length;
		if (n != b.length)
			throw new RuntimeException();
		double[] mx = new double[n];
		for (int i = 0; i < n; i++)
			mx[i] = a[i] < b[i] ? a[i] : b[i];
		return mx;
	}

	public static final double[] range(double[] e) {
		int n = e.length;
		double mx = e[0], mn = e[0];
		for (int i = 1; i < n; i++) {
			if (e[i] > mx) mx = e[i];
			else if (e[i] < mn) mn = e[i];
		}
		return new double[] { mn, mx };
	}

	public static final double iqr(double[] e) {
		double[] v = quantile0(e, new double[] {0.75, 0.25});
		return v[0] - v[1];
	}

	public static final boolean isEqual(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b) <= tol);
	}

	public static final boolean isEqualScaled(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b)/(Double.isNaN(a) ? 0 : a) <= tol);
	}

	public static final boolean allEqual(double[] a, double[] b, double tol) {
		int n = a.length;
		if (n != b.length) throw new RuntimeException();
		for (int i = 0; i < n; i++)
			if (!isEqual(a[i], b[i], tol)) return false;
		return true;
	}

	public static final boolean allEqualScaled(double[] a, double[] b, double tol) {
		int n = a.length;
		if (n != b.length) throw new RuntimeException();
		for (int i = 0; i < n; i++)
			if (!isEqualScaled(a[i], b[i], tol)) return false;
		return true;
	}

	public static final boolean isSorted(double[] a, boolean ascending) {
		int n = a.length;
		if (ascending) {
			for (int i = 1; i < n; i++)
				if (a[i-1] > a[i]) return false;
		} else {
			for (int i = 1; i < n; i++)
				if (a[i-1] < a[i]) return false;
		}
		return true;
	}
}
