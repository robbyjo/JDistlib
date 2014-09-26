/*
 * Roby Joehanes
 * 
 * Copyright 2007 Roby Joehanes
 * This file is distributed under the GNU General Public License version 3.0.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package jdistlib.math;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;

/**
 * <P>Routines that I took from LINPACK library. So far, only dpbfa and dpbsl
 * are taken. I hand-translated them from FORTRAN. I don't change the method
 * names, just in case if you people are familiar with it already. Also, please
 * pay attention to the parameter names. I removed all superfluous indices.
 * 
 * <P>See: <a href="http://www.netlib.org/linpack/">http://www.netlib.org/linpack/</a>
 * 
 * @author Roby Joehanes
 *
 */
public class LinPack
{
	/**
	 * <P>dpbsl solves the double precision symmetric positive definite band system  a*x = b
	 * using the factors computed by dpbco or dpbfa.
	 * 
	 * <P>(Taken from LINPACK_D)
	 * 
	 * <P>on entry
	 * 
	 * @param abd double precision(lda, n), the output from dpbco or dpbfa.
	 * @param lda integer, the leading dimension of the array  abd
	 * @param n integer, the order of the matrix  a
	 * @param m integer, the number of diagonals above the main diagonal.
	 * @param b double precision(n), the right hand side vector.
	 * 
	 * <p>On return
	 * @return b       the solution vector  x
	 * 
	 * <Pre>error condition
	 * a division by zero will occur if the input factor contains
	 * a zero on the diagonal.  technically this indicates singularity
	 * but it is usually caused by improper subroutine arguments.
	 * it will not occur if the subroutines are called correctly and  info == 0 .
	 * 
	 * to compute  inverse(a) * c  where  c  is a matrix with  p  columns
	 * 
	 * call dpbco(abd,lda,n,rcond,z,info)
	 * if (rcond is too small .or. info .ne. 0) go to ...
	 *    do 10 j = 1, p
	 *       call dpbsl(abd,lda,n,c(1,j))
	 *    10 continue
	 * 
	 * linpack.  this version dated 08/14/78
	 * cleve moler, university of new mexico, argonne national lab.
	 * </pre>
	 * @param abd Banded matrix.
	 * @param m Band width
	 * @param b the Y vector. Holds the output after the routine ends.
	 */
	public static final void dpbsl(double[][] abd, int m, double[] b)
	{
		double t;
		int
			n = abd[0].length,
			la, lb, lm;
	
		// solve trans(r)*y = b
		for (int k = 0; k < n; ++k) {
			lm = min(k, m);
			la = m - lm;
			lb = k - lm;
			t = 0;
			for (int i = 0; i < lm; i++)
				t += abd[la + i][k] * b[lb + i];
			b[k] = (b[k] - t) / abd[m][k];
		}
	
		// solve r*x = y
		for (int kb = 0; kb < n; ++kb) {
			int k = n - kb - 1;
			lm = min(k, m);
			la = m - lm;
			lb = k - lm;
			b[k] /= abd[m][k];
			t = -b[k];
			for (int i = 0; i < lm; i++)
				b[lb + i] += t * abd[la + i][k];
		}
	}

	/**
	 * <P>dpbfa factors a double precision symmetric positive definite
	 * matrix stored in band form.
	 *     
	 * <P>dpbfa is usually called by dpbco, but it can be called
	 * directly with a saving in time if  rcond  is not needed.
	 * 
	 * <P>(Taken from LINPACK_D)
	 * 
	 * <pre>
	 * on entry
	 * abd     double precision(lda, n)
	 *         the matrix to be factored.  the columns of the upper
	 *         triangle are stored in the columns of abd and the
	 *         diagonals of the upper triangle are stored in the
	 *         rows of abd .  see the comments below for details.
	 * 
	 * lda     integer
	 *         the leading dimension of the array abd. lda must be >= m + 1
	 * 
	 * n       integer
	 *         the order of the matrix  a .
	 * 
	 * m       integer
	 *         the number of diagonals above the main diagonal. 0 <= m < n
	 * 
	 * on return
	 * abd     an upper triangular matrix  r , stored in band form, so that a = trans(r)*r .
	 * info    integer
	 *         = 0  for normal return.
	 *         = k  if the leading minor of order  k  is not positive definite.
	 * 
	 * band storage
	 * if  a  is a symmetric positive definite band matrix,
	 * the following program segment will set up the input.
	 *     m = (band width above diagonal)
	 *     do 20 j = 1, n
	 *        i1 = max0(1, j-m)
	 *        do 10 i = i1, j
	 *           k = i-j+m+1
	 *           abd(k,j) = a(i,j)
	 *        10    continue
	 *     20 continue
	 * linpack.  this version dated 08/14/78 .
	 * cleve moler, university of new mexico, argonne national lab.
	 * begin block with ...exits to 40
	 * </pre>
	 * @param abd Banded matrix
	 * @param m Band width
	 */
	public static final int dpbfa(double[][] abd, int m)
	{
		int n = abd[0].length;
	
		for (int j = 0; j < n; ++j)
		{
			double s = 0;
			for (int mu = max(m - j, 0), k = mu, ik = m, jk = max(j - m, 0); k < m; ++k, --ik, ++jk)
			{
				double sum = 0;
				int length = k - mu;
				for (int l = 0; l < length; l++)
					sum += abd[ik + l][jk] * abd[mu + l][j];
				double t = (abd[k][j] - sum) / abd[m][jk];
				abd[k][j] = t;
				s += t * t;
			}
			s = abd[m][j] - s;
			if (s <= 0)
				return j;
			abd[m][j] = sqrt(s);
		}
		return 0;
	}
}
