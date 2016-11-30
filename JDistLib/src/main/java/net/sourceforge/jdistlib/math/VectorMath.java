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
package net.sourceforge.jdistlib.math;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static net.sourceforge.jdistlib.util.Utilities.sort;

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

	public static final double[] vpow(int[] x, int[] e) {
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

	/**
	 * Vector signum.
	 * @param e
	 * @return -1 if e[i] < 0; 0 if e[i] == 0; 1 if e[i] > 0
	 */
	public static final int[] vsgn(double[] e) {
		int[] v = new int[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = e[i] < 0 ? -1 : e[i] == 0 ? 0 : 1;
		return v;
	}

	public static final int[] vsgn(int[] e) {
		int[] v = new int[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = e[i] < 0 ? -1 : e[i] == 0 ? 0 : 1;
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
	 * Euclidian distance / Root mean square
	 * @param e
	 * @return sqrt(mean(e * e))
	 */
	public static final double distance(double[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += (e[i] * e[i] / n); // guard against overflow
		return sqrt(sum);
	}

	/**
	 * Return summary statistics
	 * @param e
	 * @return an array of 6 elements: Min, Q1, Median, Mean, Q3, Max
	 */
	public static final double[] summary(double[] e) {
		int n = e.length;
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, n);
		sort(v);
		double[] s = quantile(v, new double[] {0.25, 0.5, 0.75});
		return new double[] {v[0], s[0], s[1], mean(v), s[2], v[n - 1]};
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

	public static final double sum(int[] e) {
		double sum = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			sum += e[i];
		return sum;
	}

	/**
	 * Product of numbers. Implemented as exp(sum(log(e))).
	 * @param e
	 * @return e[0] * e[1] * ... * e[n-1]
	 */
	public static final double prod(double[] e) {
		double prod = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			prod += log(e[i]);
		return exp(prod);
	}

	/**
	 * Log of product of numbers. Implemented as sum(log(e)).
	 * @param e
	 * @return log(e[0]) + log(e[1]) + ... + log(e[n-1])
	 */
	public static final double log_prod(double[] e) {
		double prod = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			prod += log(e[i]);
		return prod;
	}

	/**
	 * Geometric mean
	 * @param e
	 */
	public static final double geom_mean(double[] e) {
		double prod = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			prod += log(e[i]);
		return exp(prod/n);
	}

	/**
	 * Harmonic mean
	 * @param e
	 */
	public static final double harm_mean(double[] e) {
		double prod = 0;
		int n = e.length;
		for (int i = 0; i < n; i++)
			prod += 1.0/e[i];
		return n/prod;
	}

	/**
	 * Weighted sum / Dot product
	 * @param e
	 * @param w
	 * @return sum(e * w)
	 */
	public static final double dot(double[] e, double[] w) {
		int n = e.length;
		if (w.length != n) throw new RuntimeException();
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += e[i] * w[i];
		return sum;
	}

	/**
	 * Weighted sum / Dot product
	 * @param e
	 * @param w
	 * @return sum(e * w)
	 */
	public static final double dot(int[] e, double[] w) {
		int n = e.length;
		if (w.length != n) throw new RuntimeException();
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += e[i] * w[i];
		return sum;
	}

	/**
	 * Weighted sum / Dot product
	 * @param e
	 * @param w
	 * @return sum(e * w)
	 */
	public static final double dot(double[] e, int[] w) {
		int n = e.length;
		if (w.length != n) throw new RuntimeException();
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += e[i] * w[i];
		return sum;
	}

	/**
	 * Weighted sum / Dot product
	 * @param e
	 * @param w
	 * @return sum(e * w)
	 */
	public static final double dot(int[] e, int[] w) {
		int n = e.length;
		if (w.length != n) throw new RuntimeException();
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += e[i] * w[i];
		return sum;
	}

	/**
	 * Trimmed mean of values. Lower and upper are the percentile. For example: trimmed_mean(e, 0.25, 0.75) means
	 * average all values between 0.25 and 0.75 percentile inclusive.
	 * @param e
	 * @param lower must be between 0 and 1
	 * @param upper must be between 0 and 1
	 * @return trimmed mean.
	 */
	public static final double trimmed_mean(double[] e, double lower, double upper) {
		if (lower < 0 || lower > 1 || upper < 0 || upper > 1 || lower > upper) throw new RuntimeException();
		double[] q = quantile0(e, new double[] {lower, upper});
		lower = q[0]; upper = q[1];
		int n = e.length, nn = 0;
		double sum = 0;
		for (int i = 0; i < n; i++) {
			double v = e[i];
			if (v >= lower || v <= upper) {
				sum += v;
				nn++;
			}
		}
		return sum / nn;
	}

	/**
	 * Winsorized mean of values. Lower and upper are the percentile. For example: winsor_mean(e, 0.25, 0.75) means
	 * average all values between 0.25 and 0.75 percentile inclusive, the rest will be replaced by the boundary value (i.e.,
	 * values lower than 0.25 percentile will be capped at 0.25 percentile and values higher than 0.75 percentile will be capped
	 * at 0.75 percentile).
	 * @param e
	 * @param lower must be between 0 and 1
	 * @param upper must be between 0 and 1
	 * @return winsorized mean.
	 */
	public static final double winsor_mean(double[] e, double lower, double upper) {
		if (lower < 0 || lower > 1 || upper < 0 || upper > 1 || lower > upper) throw new RuntimeException();
		double[] q = quantile0(e, new double[] {lower, upper});
		lower = q[0]; upper = q[1];
		int n = e.length;
		double sum = 0;
		for (int i = 0; i < n; i++) {
			double v = e[i];
			sum += v < lower ? lower : (v > upper ? upper : v);
		}
		return sum / n;
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
	 * Compute the Median Absolute Deviation (MAD) (i.e., median(abs(e - median(e))))
	 * @param e does not need to be sorted
	 * @return MAD value
	 */
	public static final double mad(double[] e) {
		int n = e.length;
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, n);
		sort(v);
		double med = quantile(v, 0.5);
		for (int i = 0; i < n; i++)
			v[i] = abs(v[i] - med);
		return quantile(v, 0.5);
	}

	/**
	 * Average Absolute Deviation (AAD) (i.e., mean(abs(e - median(e))))
	 * @param e does not need to be sorted
	 * @return AAD value
	 */
	public static final double aad(double[] e) {
		int n = e.length;
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, n);
		sort(v);
		double
			med = quantile(v, 0.5),
			sum = 0;
		for (int i = 0; i < n; i++)
			sum += abs(v[i] - med) / n;
		return sum;
	}

	/**
	 * Maximum deviation (i.e., max(abs(e - median(e))))
	 * @param e
	 */
	public static final double maxdev(double[] e) {
		int n = e.length;
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, n);
		sort(v);
		double
			med = quantile(v, 0.5),
			max = 0;
		for (int i = 0; i < n; i++) {
			double m = abs(v[i] - med);
			if (m > max) max = m;
		}
		return max;
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

	/**
	 * Mid-range of e (i.e., (max(e) + min(e)) / 2)
	 * @param e
	 */
	public static final double midrange(double[] e) {
		int n = e.length;
		double mx = e[0], mn = e[0];
		for (int i = 1; i < n; i++) {
			if (e[i] > mx) mx = e[i];
			else if (e[i] < mn) mn = e[i];
		}
		return (mn + mx) / 2.0;
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

	public static final int which_max(int[] e) {
		int n = e.length, which = 0;
		int mx = e[0];
		for (int i = 1; i < n; i++)
			if (e[i] > mx) { mx = e[i]; which = i; };
		return which;
	}

	public static final int which_min(int[] e) {
		int n = e.length, which = 0;
		int mn = e[0];
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

	/**
	 * Inter-quartile range (i.e., Q3 - Q1)
	 * @param e
	 */
	public static final double iqr(double[] e) {
		double[] v = quantile0(e, new double[] {0.75, 0.25});
		return v[0] - v[1];
	}

	/**
	 * Mid hinge. (Q1 + Q3) / 2.0
	 * @param e
	 */
	public static final double midhinge(double[] e) {
		double[] v = quantile0(e, new double[] {0.25, 0.75});
		return (v[0] + v[1]) / 2.0;
	}

	/**
	 * Trimean (Q1 + 2Q2 + Q3) / 4
	 * @param e
	 */
	public static final double trimean(double[] e) {
		double[] v = quantile0(e, new double[] {0.25, 0.5, 0.75});
		return (v[0] + 2*v[1] + v[2]) / 4.0;
	}

	public static final boolean isEqual(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b) <= tol);
	}

	public static final boolean isEqualScaled(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b)/(Double.isNaN(a) ? 0 : a) <= tol);
	}

	public static final double relativeDiff(double[] a, double[] b) {
		int n = a.length;
		if (n != b.length) throw new RuntimeException();
		double
			xy = mean(vabs(vmin(a, b))),
			xn = mean(vabs(a));
		return xy / xn;
	}

	public static final boolean allEqual(double[] a, double[] b, double tol) {
		return relativeDiff(a, b) < tol;
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
