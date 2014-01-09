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

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

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
			v[i] = MathFunctions.signif(-e[i], digits);
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
	 * @param sortedData
	 * @param quantile must be 0 <= quantile <= 1
	 * @return
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

	public static final double[] quantile(double[] sortedData, double[] quantile) {
		double[] v = new double[quantile.length];
		for (int i = 0; i < quantile.length; i++)
			v[i] = quantile(sortedData, quantile[i]);
		return v;
	}

	public static final double mean(double[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += (e[i] / n); // guard against overflow
		return sum;
	}

	public static final double sd(double[] e) {
		double sum = 0, sumsq = 0;
		int n = e.length;
		for (int i = 0; i < n; i++) {
			double v = e[i] / n; // guard against overflow
			sum += v;
			sumsq += v * v;
		}
		return sqrt((n * sumsq - sum * sum) * (n / (n - 1)));
	}

	public static final double var(double[] e) {
		double sum = 0, sumsq = 0;
		int n = e.length;
		for (int i = 0; i < n; i++) {
			double v = e[i] / n; // guard against overflow
			sum += v;
			sumsq += v * v;
		}
		return (n * sumsq - sum * sum) * (n / (n - 1));
	}

	public static final double sum(double[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += e[i];
		return sum;
	}

	public static final double sum(Map<String, Integer> e) {
		double sum = 0;
		for (int v: e.values())
			sum += v;
		return sum;
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
}
