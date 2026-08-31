/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import java.util.Map;
import java.util.TreeMap;

import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Deterministic reference implementations for the extended BLAS/LAPACK surface. */
final class CpuAdvancedLinearAlgebra {
	private CpuAdvancedLinearAlgebra() {}

	static void dscal(int count, double alpha, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); for (int i = 0, p = offset; i < count; i++, p += stride) x[p] *= alpha;
	}
	static void dcopy(int count, double[] x, int xo, int xs, double[] y, int yo, int ys) {
		checkRegion(count, x, xo, xs); checkRegion(count, y, yo, ys);
		if (x == y) { double[] copy = new double[count]; for (int i = 0, p = xo; i < count; i++, p += xs) copy[i] = x[p];
			for (int i = 0, p = yo; i < count; i++, p += ys) y[p] = copy[i]; }
		else for (int i = 0, xp = xo, yp = yo; i < count; i++, xp += xs, yp += ys) y[yp] = x[xp];
	}
	static void dswap(int count, double[] x, int xo, int xs, double[] y, int yo, int ys) {
		checkRegion(count, x, xo, xs); checkRegion(count, y, yo, ys);
		for (int i = 0, xp = xo, yp = yo; i < count; i++, xp += xs, yp += ys) {
			double value = x[xp]; x[xp] = y[yp]; y[yp] = value;
		}
	}
	static double dasum(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); double value = 0.0;
		for (int i = 0, p = offset; i < count; i++, p += stride) value += Math.abs(x[p]); return value;
	}
	static int idamax(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return -1; int selected = 0, p = offset;
		double maximum = Math.abs(x[p]); for (int i = 1; i < count; i++) { p += stride;
			if (Math.abs(x[p]) > maximum) { maximum = Math.abs(x[p]); selected = i; } } return selected;
	}

	static void dgemv(MatrixTranspose transpose, int rows, int columns, double alpha,
			double[] matrix, int matrixOffset, int leadingDimension, double[] x, int xOffset,
			int xStride, double beta, double[] y, int yOffset, int yStride) {
		requireTranspose(transpose); checkMatrixRegion(rows, columns, matrix, matrixOffset, leadingDimension);
		int input = transpose == MatrixTranspose.NONE ? columns : rows;
		int output = transpose == MatrixTranspose.NONE ? rows : columns;
		checkRegion(input, x, xOffset, xStride); checkRegion(output, y, yOffset, yStride);
		for (int row = 0, yp = yOffset; row < output; row++, yp += yStride) {
			double sum = 0.0;
			for (int column = 0, xp = xOffset; column < input; column++, xp += xStride)
				sum += (transpose == MatrixTranspose.NONE
						? matrix[matrixOffset + row * leadingDimension + column]
						: matrix[matrixOffset + column * leadingDimension + row]) * x[xp];
			y[yp] = alpha * sum + beta * y[yp];
		}
	}
	static void dgemm(MatrixTranspose ta, MatrixTranspose tb, int rows, int columns,
			int shared, double alpha, double[] left, int leftOffset, int leftLd,
			double[] right, int rightOffset, int rightLd, double beta, double[] result,
			int resultOffset, int resultLd) {
		requireTranspose(ta); requireTranspose(tb);
		checkMatrixRegion(ta == MatrixTranspose.NONE ? rows : shared,
				ta == MatrixTranspose.NONE ? shared : rows, left, leftOffset, leftLd);
		checkMatrixRegion(tb == MatrixTranspose.NONE ? shared : columns,
				tb == MatrixTranspose.NONE ? columns : shared, right, rightOffset, rightLd);
		checkMatrixRegion(rows, columns, result, resultOffset, resultLd);
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
			double sum = 0.0; for (int k = 0; k < shared; k++) {
				double a = ta == MatrixTranspose.NONE ? left[leftOffset + row * leftLd + k]
						: left[leftOffset + k * leftLd + row];
				double b = tb == MatrixTranspose.NONE ? right[rightOffset + k * rightLd + column]
						: right[rightOffset + column * rightLd + k]; sum += a * b;
			}
			int p = resultOffset + row * resultLd + column; result[p] = alpha * sum + beta * result[p];
		}
	}
	static void dsyrk(MatrixTranspose transpose, int dimension, int shared, double alpha,
			double[] matrix, int matrixOffset, int matrixLd, double beta, double[] result,
			int resultOffset, int resultLd) {
		requireTranspose(transpose); checkMatrixRegion(transpose == MatrixTranspose.NONE ? dimension : shared,
				transpose == MatrixTranspose.NONE ? shared : dimension, matrix, matrixOffset, matrixLd);
		checkMatrixRegion(dimension, dimension, result, resultOffset, resultLd);
		for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
			double sum = 0.0; for (int k = 0; k < shared; k++) {
				double a = transpose == MatrixTranspose.NONE ? matrix[matrixOffset + row * matrixLd + k]
						: matrix[matrixOffset + k * matrixLd + row];
				double b = transpose == MatrixTranspose.NONE ? matrix[matrixOffset + column * matrixLd + k]
						: matrix[matrixOffset + k * matrixLd + column]; sum += a * b;
			}
			double value = alpha * sum + beta * result[resultOffset + row * resultLd + column];
			result[resultOffset + row * resultLd + column] = value;
			result[resultOffset + column * resultLd + row] = value;
		}
	}
	static void dtrsm(MatrixSide side, MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int rows, int columns, double alpha, double[] matrix,
			int matrixOffset, int matrixLd, double[] right, int rightOffset, int rightLd) {
		if (side == null || triangle == null || transpose == null || diagonal == null)
			throw new IllegalArgumentException("triangular policies are required");
		int order = side == MatrixSide.LEFT ? rows : columns;
		checkMatrixRegion(order, order, matrix, matrixOffset, matrixLd);
		checkMatrixRegion(rows, columns, right, rightOffset, rightLd);
		if (side == MatrixSide.LEFT) {
			for (int column = 0; column < columns; column++) {
				double[] vector = new double[rows]; for (int row = 0; row < rows; row++)
					vector[row] = alpha * right[rightOffset + row * rightLd + column];
				dtrsvRegion(triangle, transpose, diagonal, rows, matrix, matrixOffset, matrixLd, vector);
				for (int row = 0; row < rows; row++) right[rightOffset + row * rightLd + column] = vector[row];
			}
		} else {
			MatrixTranspose reversed = transpose == MatrixTranspose.NONE ? MatrixTranspose.TRANSPOSE : MatrixTranspose.NONE;
			for (int row = 0; row < rows; row++) {
				double[] vector = new double[columns]; for (int column = 0; column < columns; column++)
					vector[column] = alpha * right[rightOffset + row * rightLd + column];
				dtrsvRegion(triangle, reversed, diagonal, columns, matrix, matrixOffset, matrixLd, vector);
				for (int column = 0; column < columns; column++) right[rightOffset + row * rightLd + column] = vector[column];
			}
		}
	}

	static void dger(int rows, int columns, double alpha, double[] x, int xo, int xs,
			double[] y, int yo, int ys, double[] matrix) {
		checkRegion(rows, x, xo, xs); checkRegion(columns, y, yo, ys); checkMatrix(rows, columns, matrix);
		for (int row = 0, xp = xo; row < rows; row++, xp += xs) for (int column = 0, yp = yo;
				column < columns; column++, yp += ys) matrix[row * columns + column] += alpha * x[xp] * y[yp];
	}
	static void dsyr(MatrixTriangle triangle, int dimension, double alpha, double[] x,
			int offset, int stride, double[] matrix) {
		checkTriangle(triangle); checkRegion(dimension, x, offset, stride); checkMatrix(dimension, dimension, matrix);
		for (int row = 0, rp = offset; row < dimension; row++, rp += stride)
			for (int column = 0, cp = offset; column <= row; column++, cp += stride) {
				int primary = triangle == MatrixTriangle.LOWER ? row * dimension + column
						: column * dimension + row;
				double updated = matrix[primary] + alpha * x[rp] * x[cp];
				matrix[row * dimension + column] = matrix[column * dimension + row] = updated;
			}
	}
	static void dsyr2(MatrixTriangle triangle, int dimension, double alpha, double[] x,
			int xo, int xs, double[] y, int yo, int ys, double[] matrix) {
		checkTriangle(triangle); checkRegion(dimension, x, xo, xs); checkRegion(dimension, y, yo, ys);
		checkMatrix(dimension, dimension, matrix);
		for (int row = 0, xp = xo, yp = yo; row < dimension; row++, xp += xs, yp += ys)
			for (int column = 0, xq = xo, yq = yo; column <= row; column++, xq += xs, yq += ys) {
				int primary = triangle == MatrixTriangle.LOWER ? row * dimension + column
						: column * dimension + row;
				double updated = matrix[primary] + alpha * (x[xp] * y[yq] + y[yp] * x[xq]);
				matrix[row * dimension + column] = matrix[column * dimension + row] = updated;
			}
	}
	static void dsymm(MatrixSide side, MatrixTriangle triangle, int rows, int columns,
			double alpha, double[] symmetric, double[] right, double beta, double[] result) {
		if (side == null) throw new IllegalArgumentException("SYMM side is required"); checkTriangle(triangle);
		int order = side == MatrixSide.LEFT ? rows : columns; checkMatrix(order, order, symmetric);
		checkMatrix(rows, columns, right); checkMatrix(rows, columns, result);
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
			double sum = 0.0; if (side == MatrixSide.LEFT) for (int k = 0; k < rows; k++)
				sum += symmetric(symmetric, order, triangle, row, k) * right[k * columns + column];
			else for (int k = 0; k < columns; k++) sum += right[row * columns + k]
					* symmetric(symmetric, order, triangle, k, column);
			int p = row * columns + column; result[p] = alpha * sum + beta * result[p];
		}
	}
	static void dsyr2k(MatrixTriangle triangle, MatrixTranspose transpose, int dimension,
			int shared, double alpha, double[] left, double[] right, double beta, double[] result) {
		checkTriangle(triangle); requireTranspose(transpose);
		int sourceRows = transpose == MatrixTranspose.NONE ? dimension : shared;
		int sourceColumns = transpose == MatrixTranspose.NONE ? shared : dimension;
		checkMatrix(sourceRows, sourceColumns, left); checkMatrix(sourceRows, sourceColumns, right);
		checkMatrix(dimension, dimension, result);
		for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
			double sum = 0.0; for (int k = 0; k < shared; k++) {
				double ar = transpose == MatrixTranspose.NONE ? left[row * shared + k] : left[k * dimension + row];
				double ac = transpose == MatrixTranspose.NONE ? left[column * shared + k] : left[k * dimension + column];
				double br = transpose == MatrixTranspose.NONE ? right[row * shared + k] : right[k * dimension + row];
				double bc = transpose == MatrixTranspose.NONE ? right[column * shared + k] : right[k * dimension + column];
				sum += ar * bc + br * ac;
			}
			int primary = triangle == MatrixTriangle.LOWER ? row * dimension + column
					: column * dimension + row;
			double value = alpha * sum + beta * result[primary];
			result[row * dimension + column] = result[column * dimension + row] = value;
		}
	}

	static LuFactor dgetrf(double[] matrix, int dimension) {
		checkMatrix(dimension, dimension, matrix); checkFinite(matrix); double[] lu = matrix.clone();
		int[] permutation = identityPermutation(dimension); int sign = 1;
		double scale = maximumAbsolute(lu), tolerance = Math.ulp(1.0) * Math.max(1.0, scale) * dimension;
		for (int k = 0; k < dimension; k++) {
			int pivot = k; for (int row = k + 1; row < dimension; row++)
				if (Math.abs(lu[row * dimension + k]) > Math.abs(lu[pivot * dimension + k])) pivot = row;
			if (!(Math.abs(lu[pivot * dimension + k]) > tolerance))
				throw new IllegalArgumentException("matrix is singular at LU column " + (k + 1));
			if (pivot != k) { swapRows(lu, dimension, k, pivot); int p = permutation[k];
				permutation[k] = permutation[pivot]; permutation[pivot] = p; sign = -sign; }
			for (int row = k + 1; row < dimension; row++) { int base = row * dimension;
				lu[base + k] /= lu[k * dimension + k]; double multiplier = lu[base + k];
				for (int column = k + 1; column < dimension; column++)
					lu[base + column] -= multiplier * lu[k * dimension + column]; }
		}
		return new LuFactor(dimension, lu, permutation, sign);
	}

	static SymmetricIndefiniteFactor dsytrf(double[] matrix, int dimension) {
		checkMatrix(dimension, dimension, matrix); checkFinite(matrix); checkSymmetric(matrix, dimension);
		double[] work = matrix.clone(), lower = identity(dimension), diagonal = new double[matrix.length];
		int[] permutation = identityPermutation(dimension), blocks = new int[dimension];
		double tolerance = Math.ulp(1.0) * Math.max(1.0, maximumAbsolute(work)) * dimension;
		for (int k = 0; k < dimension;) {
			int pivot = k; for (int i = k + 1; i < dimension; i++)
				if (Math.abs(work[i * dimension + i]) > Math.abs(work[pivot * dimension + pivot])) pivot = i;
			int pairFirst = k, pairSecond = k + 1; double maximumOffDiagonal = 0.0;
			for (int i = k; i < dimension - 1; i++) for (int j = i + 1; j < dimension; j++)
				if (Math.abs(work[i * dimension + j]) > maximumOffDiagonal) {
					maximumOffDiagonal = Math.abs(work[i * dimension + j]); pairFirst = i; pairSecond = j;
				}
			double maximumDiagonal = Math.abs(work[pivot * dimension + pivot]);
			if (maximumDiagonal > tolerance && (maximumOffDiagonal <= tolerance
					|| maximumDiagonal >= 0.6403882032022076 * maximumOffDiagonal)) {
				symmetricSwap(work, lower, permutation, dimension, k, pivot, k);
				double d = work[k * dimension + k]; diagonal[k * dimension + k] = d; blocks[k] = 1;
				for (int i = k + 1; i < dimension; i++) lower[i * dimension + k] = work[i * dimension + k] / d;
				for (int i = k + 1; i < dimension; i++) for (int j = k + 1; j <= i; j++) {
					double value = work[i * dimension + j] - lower[i * dimension + k] * d * lower[j * dimension + k];
					work[i * dimension + j] = work[j * dimension + i] = value;
				}
				k++;
			} else {
				if (k + 1 >= dimension) throw new IllegalArgumentException("symmetric matrix is singular at pivot " + (k + 1));
				int first = pairFirst, second = pairSecond;
				if (!(maximumOffDiagonal > tolerance)) throw new IllegalArgumentException("symmetric matrix is singular at pivot " + (k + 1));
				symmetricSwap(work, lower, permutation, dimension, k, first, k);
				if (second == k) second = first; else if (second == first) second = k;
				symmetricSwap(work, lower, permutation, dimension, k + 1, second, k);
				double a = work[k * dimension + k], b = work[(k + 1) * dimension + k];
				double c = work[(k + 1) * dimension + k + 1], determinant = a * c - b * b;
				if (!(Math.abs(determinant) > tolerance * tolerance))
					throw new IllegalArgumentException("symmetric matrix has singular 2x2 pivot at " + (k + 1));
				diagonal[k * dimension + k] = a; diagonal[k * dimension + k + 1] = b;
				diagonal[(k + 1) * dimension + k] = b; diagonal[(k + 1) * dimension + k + 1] = c;
				blocks[k] = 2; blocks[k + 1] = 0;
				for (int i = k + 2; i < dimension; i++) {
					double firstValue = work[i * dimension + k], secondValue = work[i * dimension + k + 1];
					lower[i * dimension + k] = (firstValue * c - secondValue * b) / determinant;
					lower[i * dimension + k + 1] = (secondValue * a - firstValue * b) / determinant;
				}
				for (int i = k + 2; i < dimension; i++) for (int j = k + 2; j <= i; j++) {
					double li1 = lower[i * dimension + k], li2 = lower[i * dimension + k + 1];
					double lj1 = lower[j * dimension + k], lj2 = lower[j * dimension + k + 1];
					double correction = li1 * (a * lj1 + b * lj2) + li2 * (b * lj1 + c * lj2);
					double value = work[i * dimension + j] - correction;
					work[i * dimension + j] = work[j * dimension + i] = value;
				}
				k += 2;
			}
		}
		return new SymmetricIndefiniteFactor(dimension, lower, diagonal, permutation, blocks);
	}

	static SymmetricEigenDecomposition dsygvd(double[] matrix, double[] metric, int dimension) {
		checkMatrix(dimension, dimension, matrix); checkMatrix(dimension, dimension, metric);
		checkFinite(matrix); checkFinite(metric); checkSymmetric(matrix, dimension); checkSymmetric(metric, dimension);
		double[] lower = CpuLinearAlgebra.dpotrf(metric, dimension).lower();
		double[] transformed = matrix.clone();
		// C = inv(L) * A * inv(L')
		for (int column = 0; column < dimension; column++) {
			double[] vector = new double[dimension]; for (int row = 0; row < dimension; row++) vector[row] = transformed[row * dimension + column];
			dtrsvRegion(MatrixTriangle.LOWER, MatrixTranspose.NONE, MatrixDiagonal.NON_UNIT,
					dimension, lower, 0, dimension, vector);
			for (int row = 0; row < dimension; row++) transformed[row * dimension + column] = vector[row];
		}
		for (int row = 0; row < dimension; row++) {
			double[] vector = new double[dimension]; System.arraycopy(transformed, row * dimension, vector, 0, dimension);
			dtrsvRegion(MatrixTriangle.LOWER, MatrixTranspose.NONE, MatrixDiagonal.NON_UNIT,
					dimension, lower, 0, dimension, vector);
			System.arraycopy(vector, 0, transformed, row * dimension, dimension);
		}
		for (int row = 0; row < dimension; row++) for (int column = row + 1; column < dimension; column++) {
			double value = 0.5 * (transformed[row * dimension + column] + transformed[column * dimension + row]);
			transformed[row * dimension + column] = transformed[column * dimension + row] = value;
		}
		SymmetricEigenDecomposition ordinary = CpuLinearAlgebra.dsyev(transformed, dimension);
		double[] vectors = ordinary.eigenvectors();
		for (int column = 0; column < dimension; column++) {
			double[] vector = new double[dimension]; for (int row = 0; row < dimension; row++) vector[row] = vectors[row * dimension + column];
			dtrsvRegion(MatrixTriangle.LOWER, MatrixTranspose.TRANSPOSE, MatrixDiagonal.NON_UNIT,
					dimension, lower, 0, dimension, vector);
			for (int row = 0; row < dimension; row++) vectors[row * dimension + column] = vector[row];
		}
		return new SymmetricEigenDecomposition(dimension, ordinary.eigenvalues(), vectors);
	}

	static CsrMatrix dcsrgemm(CsrMatrix left, CsrMatrix right) {
		if (left == null || right == null || left.columns() != right.rows())
			throw new IllegalArgumentException("CSR product dimensions do not conform");
		double[] av = left.values(), bv = right.values(); int[] ac = left.columnIndices(), bc = right.columnIndices();
		int[] as = left.rowStarts(), bs = right.rowStarts();
		@SuppressWarnings({"unchecked", "rawtypes"}) Map<Integer, Double>[] rows = new Map[left.rows()]; int count = 0;
		for (int row = 0; row < left.rows(); row++) { Map<Integer, Double> values = new TreeMap<Integer, Double>();
			for (int ap = as[row] - 1; ap < as[row + 1] - 1; ap++) { int middle = ac[ap] - 1;
				for (int bp = bs[middle] - 1; bp < bs[middle + 1] - 1; bp++) { int column = bc[bp];
					Double old = values.get(column); values.put(column, (old == null ? 0.0 : old) + av[ap] * bv[bp]); } }
			values.values().removeIf(value -> value == 0.0); rows[row] = values; count += values.size(); }
		double[] values = new double[count]; int[] columns = new int[count], starts = new int[left.rows() + 1];
		starts[0] = 1; int offset = 0; for (int row = 0; row < left.rows(); row++) {
			for (Map.Entry<Integer, Double> entry : rows[row].entrySet()) { columns[offset] = entry.getKey(); values[offset++] = entry.getValue(); }
			starts[row + 1] = offset + 1; }
		return new CsrMatrix(left.rows(), right.columns(), values, columns, starts);
	}
	static void dcsrsv(MatrixTriangle triangle, MatrixTranspose transpose, MatrixDiagonal diagonal,
			CsrMatrix matrix, double[] vector) {
		checkTriangle(triangle); requireTranspose(transpose);
		if (diagonal == null || matrix == null || matrix.rows() != matrix.columns()
				|| vector == null || vector.length != matrix.rows())
			throw new IllegalArgumentException("CSR triangular solve dimensions do not conform");
		CsrMatrix source = transpose == MatrixTranspose.NONE ? matrix : transpose(matrix);
		MatrixTriangle effective = transpose == MatrixTranspose.NONE ? triangle
				: triangle == MatrixTriangle.LOWER ? MatrixTriangle.UPPER : MatrixTriangle.LOWER;
		double[] values = source.values(); int[] columns = source.columnIndices(), starts = source.rowStarts();
		int begin = effective == MatrixTriangle.LOWER ? 0 : source.rows() - 1;
		int end = effective == MatrixTriangle.LOWER ? source.rows() : -1;
		int step = effective == MatrixTriangle.LOWER ? 1 : -1;
		for (int row = begin; row != end; row += step) { double value = vector[row];
			double pivot = diagonal == MatrixDiagonal.UNIT ? 1.0 : 0.0;
			for (int p = starts[row] - 1; p < starts[row + 1] - 1; p++) { int column = columns[p] - 1;
				if (column == row && diagonal == MatrixDiagonal.NON_UNIT) pivot += values[p];
				else if ((effective == MatrixTriangle.LOWER && column < row)
						|| (effective == MatrixTriangle.UPPER && column > row)) value -= values[p] * vector[column]; }
			if (diagonal == MatrixDiagonal.NON_UNIT) { if (pivot == 0.0) throw new IllegalArgumentException("zero CSR triangular diagonal"); value /= pivot; }
			vector[row] = value; }
	}

	static PreparedDenseMatrix prepareDge(double[] matrix, int rows, int columns) {
		checkMatrix(rows, columns, matrix); final double[] retained = matrix.clone();
		return new PreparedDenseMatrix() { private boolean closed;
			@Override public int rows() { checkOpen(); return rows; }
			@Override public int columns() { checkOpen(); return columns; }
			@Override public void multiply(MatrixTranspose transpose, double alpha, double[] right,
					int rightColumns, double beta, double[] result) { checkOpen(); int outputRows = transpose == MatrixTranspose.NONE ? rows : columns;
				int shared = transpose == MatrixTranspose.NONE ? columns : rows;
				CpuLinearAlgebra.dgemm(transpose, MatrixTranspose.NONE, outputRows, rightColumns,
						shared, alpha, retained, right, beta, result); }
			@Override public void close() { closed = true; }
			private void checkOpen() { if (closed) throw new IllegalStateException("prepared dense matrix is closed"); }
		};
	}

	// FP32 counterparts deliberately preserve FP32 arithmetic for BLAS operations.
	static void sscal(int n, float a, float[] x, int o, int s) { checkRegion(n,x,o,s); for(int i=0,p=o;i<n;i++,p+=s)x[p]*=a; }
	static void scopy(int n,float[]x,int xo,int xs,float[]y,int yo,int ys){checkRegion(n,x,xo,xs);checkRegion(n,y,yo,ys);float[]tmp=x==y?new float[n]:null;for(int i=0,p=xo;i<n;i++,p+=xs){if(tmp!=null)tmp[i]=x[p];else y[yo+i*ys]=x[p];}if(tmp!=null)for(int i=0,p=yo;i<n;i++,p+=ys)y[p]=tmp[i];}
	static void sswap(int n,float[]x,int xo,int xs,float[]y,int yo,int ys){checkRegion(n,x,xo,xs);checkRegion(n,y,yo,ys);for(int i=0,xp=xo,yp=yo;i<n;i++,xp+=xs,yp+=ys){float v=x[xp];x[xp]=y[yp];y[yp]=v;}}
	static float sasum(int n,float[]x,int o,int s){checkRegion(n,x,o,s);float v=0;for(int i=0,p=o;i<n;i++,p+=s)v+=Math.abs(x[p]);return v;}
	static int isamax(int n,float[]x,int o,int s){checkRegion(n,x,o,s);if(n==0)return-1;int q=0,p=o;float m=Math.abs(x[p]);for(int i=1;i<n;i++){p+=s;if(Math.abs(x[p])>m){m=Math.abs(x[p]);q=i;}}return q;}
	static void sgemv(MatrixTranspose t,int rows,int columns,float a,float[]m,int mo,int ld,float[]x,int xo,int xs,float b,float[]y,int yo,int ys){requireTranspose(t);checkMatrixRegion(rows,columns,m,mo,ld);int input=t==MatrixTranspose.NONE?columns:rows,output=t==MatrixTranspose.NONE?rows:columns;checkRegion(input,x,xo,xs);checkRegion(output,y,yo,ys);for(int row=0,yp=yo;row<output;row++,yp+=ys){float sum=0;for(int column=0,xp=xo;column<input;column++,xp+=xs)sum+=(t==MatrixTranspose.NONE?m[mo+row*ld+column]:m[mo+column*ld+row])*x[xp];y[yp]=a*sum+b*y[yp];}}
	static void sgemm(MatrixTranspose ta,MatrixTranspose tb,int rows,int columns,int shared,float a,float[]left,int lo,int lld,float[]right,int ro,int rld,float b,float[]result,int zo,int zld){requireTranspose(ta);requireTranspose(tb);checkMatrixRegion(ta==MatrixTranspose.NONE?rows:shared,ta==MatrixTranspose.NONE?shared:rows,left,lo,lld);checkMatrixRegion(tb==MatrixTranspose.NONE?shared:columns,tb==MatrixTranspose.NONE?columns:shared,right,ro,rld);checkMatrixRegion(rows,columns,result,zo,zld);for(int row=0;row<rows;row++)for(int column=0;column<columns;column++){float sum=0;for(int k=0;k<shared;k++)sum+=(ta==MatrixTranspose.NONE?left[lo+row*lld+k]:left[lo+k*lld+row])*(tb==MatrixTranspose.NONE?right[ro+k*rld+column]:right[ro+column*rld+k]);int p=zo+row*zld+column;result[p]=a*sum+b*result[p];}}
	static void ssyrk(MatrixTranspose t,int n,int shared,float a,float[]m,int mo,int mld,float b,float[]z,int zo,int zld){requireTranspose(t);checkMatrixRegion(t==MatrixTranspose.NONE?n:shared,t==MatrixTranspose.NONE?shared:n,m,mo,mld);checkMatrixRegion(n,n,z,zo,zld);for(int row=0;row<n;row++)for(int column=0;column<=row;column++){float sum=0;for(int k=0;k<shared;k++)sum+=(t==MatrixTranspose.NONE?m[mo+row*mld+k]:m[mo+k*mld+row])*(t==MatrixTranspose.NONE?m[mo+column*mld+k]:m[mo+k*mld+column]);float value=a*sum+b*z[zo+row*zld+column];z[zo+row*zld+column]=z[zo+column*zld+row]=value;}}
	static void strsm(MatrixSide side,MatrixTriangle triangle,MatrixTranspose t,MatrixDiagonal d,int rows,int columns,float a,float[]m,int mo,int mld,float[]q,int qo,int qld){if(side==null||triangle==null||t==null||d==null)throw new IllegalArgumentException("FP32 triangular policies are required");int order=side==MatrixSide.LEFT?rows:columns;checkMatrixRegion(order,order,m,mo,mld);checkMatrixRegion(rows,columns,q,qo,qld);if(side==MatrixSide.LEFT){for(int column=0;column<columns;column++){float[]v=new float[rows];for(int row=0;row<rows;row++)v[row]=a*q[qo+row*qld+column];strsvRegion(triangle,t,d,rows,m,mo,mld,v);for(int row=0;row<rows;row++)q[qo+row*qld+column]=v[row];}}else{MatrixTranspose reversed=t==MatrixTranspose.NONE?MatrixTranspose.TRANSPOSE:MatrixTranspose.NONE;for(int row=0;row<rows;row++){float[]v=new float[columns];for(int column=0;column<columns;column++)v[column]=a*q[qo+row*qld+column];strsvRegion(triangle,reversed,d,columns,m,mo,mld,v);for(int column=0;column<columns;column++)q[qo+row*qld+column]=v[column];}}}
	static void sger(int rows,int columns,float a,float[]x,int xo,int xs,float[]y,int yo,int ys,float[]m){checkRegion(rows,x,xo,xs);checkRegion(columns,y,yo,ys);checkMatrix(rows,columns,m);for(int row=0,xp=xo;row<rows;row++,xp+=xs)for(int column=0,yp=yo;column<columns;column++,yp+=ys)m[row*columns+column]+=a*x[xp]*y[yp];}
	static void ssyr(MatrixTriangle t,int n,float a,float[]x,int o,int s,float[]m){checkTriangle(t);checkRegion(n,x,o,s);checkMatrix(n,n,m);for(int row=0,rp=o;row<n;row++,rp+=s)for(int column=0,cp=o;column<=row;column++,cp+=s){int primary=t==MatrixTriangle.LOWER?row*n+column:column*n+row;float updated=m[primary]+a*x[rp]*x[cp];m[row*n+column]=m[column*n+row]=updated;}}
	static void ssyr2(MatrixTriangle t,int n,float a,float[]x,int xo,int xs,float[]y,int yo,int ys,float[]m){checkTriangle(t);checkRegion(n,x,xo,xs);checkRegion(n,y,yo,ys);checkMatrix(n,n,m);for(int row=0,xp=xo,yp=yo;row<n;row++,xp+=xs,yp+=ys)for(int column=0,xq=xo,yq=yo;column<=row;column++,xq+=xs,yq+=ys){int primary=t==MatrixTriangle.LOWER?row*n+column:column*n+row;float updated=m[primary]+a*(x[xp]*y[yq]+y[yp]*x[xq]);m[row*n+column]=m[column*n+row]=updated;}}
	static void ssymm(MatrixSide side,MatrixTriangle t,int rows,int columns,float a,float[]s,float[]q,float b,float[]z){if(side==null)throw new IllegalArgumentException("FP32 SYMM side is required");checkTriangle(t);int order=side==MatrixSide.LEFT?rows:columns;checkMatrix(order,order,s);checkMatrix(rows,columns,q);checkMatrix(rows,columns,z);for(int row=0;row<rows;row++)for(int column=0;column<columns;column++){float sum=0;if(side==MatrixSide.LEFT)for(int k=0;k<rows;k++)sum+=symmetric(s,order,t,row,k)*q[k*columns+column];else for(int k=0;k<columns;k++)sum+=q[row*columns+k]*symmetric(s,order,t,k,column);int p=row*columns+column;z[p]=a*sum+b*z[p];}}
	static void ssyr2k(MatrixTriangle triangle,MatrixTranspose t,int n,int shared,float a,float[]left,float[]right,float b,float[]z){checkTriangle(triangle);requireTranspose(t);int rows=t==MatrixTranspose.NONE?n:shared,columns=t==MatrixTranspose.NONE?shared:n;checkMatrix(rows,columns,left);checkMatrix(rows,columns,right);checkMatrix(n,n,z);for(int row=0;row<n;row++)for(int column=0;column<=row;column++){float sum=0;for(int k=0;k<shared;k++){float ar=t==MatrixTranspose.NONE?left[row*shared+k]:left[k*n+row],ac=t==MatrixTranspose.NONE?left[column*shared+k]:left[k*n+column],br=t==MatrixTranspose.NONE?right[row*shared+k]:right[k*n+row],bc=t==MatrixTranspose.NONE?right[column*shared+k]:right[k*n+column];sum+=ar*bc+br*ac;}int primary=triangle==MatrixTriangle.LOWER?row*n+column:column*n+row;float value=a*sum+b*z[primary];z[row*n+column]=z[column*n+row]=value;}}
	static FloatLuFactor sgetrf(float[] matrix,int n){double[]source=toDouble(matrix);LuFactor f=dgetrf(source,n);double[]p=f.packed();float[]packed=new float[p.length];copyToFloat(p,packed);return new FloatLuFactor(n,packed,f.permutation(),rawPermutationSign(f.permutation()));}
	static FloatSymmetricIndefiniteFactor ssytrf(float[]matrix,int n){SymmetricIndefiniteFactor f=dsytrf(toDouble(matrix),n);float[]l=new float[n*n],d=new float[n*n];copyToFloat(f.lower(),l);copyToFloat(f.diagonalBlocks(),d);return new FloatSymmetricIndefiniteFactor(n,l,d,f.permutation(),f.blockSizes());}
	static FloatSymmetricEigenDecomposition ssygvd(float[]matrix,float[]metric,int n){SymmetricEigenDecomposition f=dsygvd(toDouble(matrix),toDouble(metric),n);float[]v=new float[n],q=new float[n*n];copyToFloat(f.eigenvalues(),v);copyToFloat(f.eigenvectors(),q);return new FloatSymmetricEigenDecomposition(n,v,q);}
	static FloatCsrMatrix scsrgemm(FloatCsrMatrix left,FloatCsrMatrix right){CsrMatrix d=dcsrgemm(toDouble(left),toDouble(right));float[]v=new float[d.nonzeroCount()];copyToFloat(d.values(),v);return new FloatCsrMatrix(d.rows(),d.columns(),v,d.columnIndices(),d.rowStarts());}
	static void scsrsv(MatrixTriangle triangle,MatrixTranspose t,MatrixDiagonal d,FloatCsrMatrix matrix,float[]vector){double[]v=toDouble(vector);dcsrsv(triangle,t,d,toDouble(matrix),v);copyToFloat(v,vector);}
	static PreparedFloatDenseMatrix prepareSge(final float[]matrix,final int rows,final int columns){checkMatrix(rows,columns,matrix);final float[]retained=matrix.clone();return new PreparedFloatDenseMatrix(){private boolean closed;public int rows(){check();return rows;}public int columns(){check();return columns;}public void multiply(MatrixTranspose t,float a,float[]right,int rc,float b,float[]result){check();int output=t==MatrixTranspose.NONE?rows:columns,shared=t==MatrixTranspose.NONE?columns:rows;CpuSinglePrecisionLinearAlgebra.sgemm(t,MatrixTranspose.NONE,output,rc,shared,a,retained,right,b,result);}public void close(){closed=true;}private void check(){if(closed)throw new IllegalStateException("prepared FP32 dense matrix is closed");}};}

	private static int rawPermutationSign(int[] permutation){int inversions=0;for(int i=0;i<permutation.length;i++)for(int j=i+1;j<permutation.length;j++)if(permutation[i]>permutation[j])inversions++;return(inversions&1)==0?1:-1;}
	private static CsrMatrix toDouble(FloatCsrMatrix value){float[]source=value.values();double[]v=new double[source.length];for(int i=0;i<v.length;i++)v[i]=source[i];return new CsrMatrix(value.rows(),value.columns(),v,value.columnIndices(),value.rowStarts());}
	private static double[] toDouble(float[] value){if(value==null)return null;double[]result=new double[value.length];for(int i=0;i<value.length;i++)result[i]=value[i];return result;}
	private static void copyToFloat(double[]source,float[]target){for(int i=0;i<source.length;i++)target[i]=(float)source[i];}
	private static void checkRegion(int count,double[]v,int o,int s){if(count<0||v==null||o<0||s<1||(count>0&&(long)o+(long)(count-1)*s>=v.length))throw new IllegalArgumentException("invalid strided vector region");}
	private static void checkRegion(int count,float[]v,int o,int s){if(count<0||v==null||o<0||s<1||(count>0&&(long)o+(long)(count-1)*s>=v.length))throw new IllegalArgumentException("invalid FP32 strided vector region");}
	private static void checkMatrix(int rows,int columns,double[]m){if(rows<1||columns<1||m==null||m.length!=rows*columns)throw new IllegalArgumentException("invalid matrix dimensions");}
	private static void checkMatrix(int rows,int columns,float[]m){if(rows<1||columns<1||m==null||m.length!=rows*columns)throw new IllegalArgumentException("invalid FP32 matrix dimensions");}
	private static void checkMatrixRegion(int rows,int columns,double[]m,int o,int ld){if(rows<1||columns<1||m==null||o<0||ld<columns||(long)o+(long)(rows-1)*ld+columns>m.length)throw new IllegalArgumentException("invalid row-major matrix region");}
	private static void checkMatrixRegion(int rows,int columns,float[]m,int o,int ld){if(rows<1||columns<1||m==null||o<0||ld<columns||(long)o+(long)(rows-1)*ld+columns>m.length)throw new IllegalArgumentException("invalid FP32 row-major matrix region");}
	private static void checkTriangle(MatrixTriangle t){if(t==null)throw new IllegalArgumentException("matrix triangle is required");}
	private static void requireTranspose(MatrixTranspose t){if(t==null)throw new IllegalArgumentException("matrix transpose policy is required");}
	private static void checkFinite(double[]v){for(double x:v)if(!Double.isFinite(x))throw new IllegalArgumentException("matrix must be finite");}
	private static void checkSymmetric(double[]m,int n){double scale=Math.max(1.0,maximumAbsolute(m)),tol=64*Math.ulp(1.0)*scale;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)if(Math.abs(m[i*n+j]-m[j*n+i])>tol)throw new IllegalArgumentException("matrix must be symmetric");}
	private static double maximumAbsolute(double[]v){double m=0;for(double x:v)m=Math.max(m,Math.abs(x));return m;}
	private static int[] identityPermutation(int n){int[]p=new int[n];for(int i=0;i<n;i++)p[i]=i;return p;}
	private static double[] identity(int n){double[]v=new double[n*n];for(int i=0;i<n;i++)v[i*n+i]=1;return v;}
	private static void swapRows(double[]m,int n,int a,int b){for(int j=0;j<n;j++){double v=m[a*n+j];m[a*n+j]=m[b*n+j];m[b*n+j]=v;}}
	private static void symmetricSwap(double[]a,double[]l,int[]p,int n,int first,int second,int completed){if(first==second)return;for(int j=0;j<n;j++){double v=a[first*n+j];a[first*n+j]=a[second*n+j];a[second*n+j]=v;}for(int i=0;i<n;i++){double v=a[i*n+first];a[i*n+first]=a[i*n+second];a[i*n+second]=v;}for(int j=0;j<completed;j++){double v=l[first*n+j];l[first*n+j]=l[second*n+j];l[second*n+j]=v;}int v=p[first];p[first]=p[second];p[second]=v;}
	private static double symmetric(double[]m,int n,MatrixTriangle t,int row,int column){return t==MatrixTriangle.LOWER?(row>=column?m[row*n+column]:m[column*n+row]):(row<=column?m[row*n+column]:m[column*n+row]);}
	private static float symmetric(float[]m,int n,MatrixTriangle t,int row,int column){return t==MatrixTriangle.LOWER?(row>=column?m[row*n+column]:m[column*n+row]):(row<=column?m[row*n+column]:m[column*n+row]);}
	private static void dtrsvRegion(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int n,double[]a,int o,int ld,double[]x){boolean lower=(triangle==MatrixTriangle.LOWER)==(transpose==MatrixTranspose.NONE);if(lower){for(int i=0;i<n;i++){double v=x[i];for(int j=0;j<i;j++)v-=(transpose==MatrixTranspose.NONE?a[o+i*ld+j]:a[o+j*ld+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?v:v/a[o+i*ld+i];}}else{for(int i=n-1;i>=0;i--){double v=x[i];for(int j=i+1;j<n;j++)v-=(transpose==MatrixTranspose.NONE?a[o+i*ld+j]:a[o+j*ld+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?v:v/a[o+i*ld+i];}}}
	private static void strsvRegion(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int n,float[]a,int o,int ld,float[]x){boolean lower=(triangle==MatrixTriangle.LOWER)==(transpose==MatrixTranspose.NONE);if(lower){for(int i=0;i<n;i++){float v=x[i];for(int j=0;j<i;j++)v-=(transpose==MatrixTranspose.NONE?a[o+i*ld+j]:a[o+j*ld+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?v:v/a[o+i*ld+i];}}else{for(int i=n-1;i>=0;i--){float v=x[i];for(int j=i+1;j<n;j++)v-=(transpose==MatrixTranspose.NONE?a[o+i*ld+j]:a[o+j*ld+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?v:v/a[o+i*ld+i];}}}
	private static CsrMatrix transpose(CsrMatrix matrix){int rows=matrix.rows(),columns=matrix.columns(),nnz=matrix.nonzeroCount();double[]v=matrix.values(),out=new double[nnz];int[]c=matrix.columnIndices(),s=matrix.rowStarts(),counts=new int[columns],starts=new int[columns+1],indices=new int[nnz];for(int value:c)counts[value-1]++;starts[0]=1;for(int i=0;i<columns;i++)starts[i+1]=starts[i]+counts[i];int[]next=starts.clone();for(int row=0;row<rows;row++)for(int p=s[row]-1;p<s[row+1]-1;p++){int target=next[c[p]-1]++-1;indices[target]=row+1;out[target]=v[p];}return new CsrMatrix(columns,rows,out,indices,starts);}
}
