/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

import jdistlib.matrix.CsrMatrix;

/** Shared deterministic CPU reference algorithms for default backend methods. */
final class CpuLinearAlgebra {
	private CpuLinearAlgebra() {}

	static void daxpy(int count, double alpha, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		for (int i = 0, xi = xOffset, yi = yOffset; i < count;
				i++, xi += xStride, yi += yStride) y[yi] = alpha * x[xi] + y[yi];
	}

	static double ddot(int count, double[] x, int xOffset, int xStride,
			double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		double result = 0.0;
		for (int i = 0, xi = xOffset, yi = yOffset; i < count;
				i++, xi += xStride, yi += yStride) result += x[xi] * y[yi];
		return result;
	}

	static double dnrm2(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); double scale = 0.0, sum = 1.0;
		for (int i = 0, index = offset; i < count; i++, index += stride) {
			double value = Math.abs(x[index]);
			if (value != 0.0) {
				if (scale < value) { double ratio = scale / value; sum = 1.0 + sum * ratio * ratio; scale = value; }
				else { double ratio = value / scale; sum += ratio * ratio; }
			}
		}
		return scale == 0.0 ? 0.0 : scale * Math.sqrt(sum);
	}

	static void dgemv(MatrixTranspose transpose, int rows, int columns, double alpha,
			double[] matrix, double[] x, double beta, double[] y) {
		checkMatrix(rows, columns, matrix); requireTranspose(transpose);
		int input = transpose == MatrixTranspose.NONE ? columns : rows;
		int output = transpose == MatrixTranspose.NONE ? rows : columns;
		if (x == null || x.length != input || y == null || y.length != output)
			throw new IllegalArgumentException("GEMV vector dimensions do not conform");
		for (int row = 0; row < output; row++) {
			double sum = 0.0;
			for (int column = 0; column < input; column++)
				sum += (transpose == MatrixTranspose.NONE
						? matrix[row * columns + column]
						: matrix[column * columns + row]) * x[column];
			y[row] = alpha * sum + beta * y[row];
		}
	}

	static void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left,
			double[] right, double beta, double[] result) {
		requireTranspose(leftTranspose); requireTranspose(rightTranspose);
		int leftRows = leftTranspose == MatrixTranspose.NONE ? rows : shared;
		int leftColumns = leftTranspose == MatrixTranspose.NONE ? shared : rows;
		int rightRows = rightTranspose == MatrixTranspose.NONE ? shared : columns;
		int rightColumns = rightTranspose == MatrixTranspose.NONE ? columns : shared;
		checkMatrix(leftRows, leftColumns, left); checkMatrix(rightRows, rightColumns, right);
		if (result == null || result.length != rows * columns)
			throw new IllegalArgumentException("GEMM result dimensions do not conform");
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
			double sum = 0.0;
			for (int inner = 0; inner < shared; inner++) {
				double a = leftTranspose == MatrixTranspose.NONE
						? left[row * shared + inner] : left[inner * rows + row];
				double b = rightTranspose == MatrixTranspose.NONE
						? right[inner * columns + column] : right[column * shared + inner];
				sum += a * b;
			}
			int index = row * columns + column;
			result[index] = alpha * sum + beta * result[index];
		}
	}
	static void dsyrk(MatrixTranspose transpose,int dimension,int shared,double alpha,double[]matrix,double beta,double[]result){requireTranspose(transpose);checkMatrix(transpose==MatrixTranspose.NONE?dimension:shared,transpose==MatrixTranspose.NONE?shared:dimension,matrix);checkMatrix(dimension,dimension,result);for(int row=0;row<dimension;row++)for(int column=0;column<=row;column++){double sum=0;for(int k=0;k<shared;k++){double left=transpose==MatrixTranspose.NONE?matrix[row*shared+k]:matrix[k*dimension+row],right=transpose==MatrixTranspose.NONE?matrix[column*shared+k]:matrix[k*dimension+column];sum+=left*right;}double value=alpha*sum+beta*result[row*dimension+column];result[row*dimension+column]=result[column*dimension+row]=value;}}
	static void dtrsv(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int dimension,double[]matrix,double[]vector){checkTriangular(triangle,transpose,diagonal,dimension,matrix,vector);solveTriangular(triangle,transpose,diagonal,dimension,matrix,vector);}
	static void dtrsm(MatrixSide side,MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int rows,int columns,double alpha,double[]matrix,double[]right){if(side==null||rows<1||columns<1||right==null||right.length!=rows*columns)throw new IllegalArgumentException("invalid TRSM dimensions");int dimension=side==MatrixSide.LEFT?rows:columns;checkTriangular(triangle,transpose,diagonal,dimension,matrix,new double[dimension]);if(side==MatrixSide.LEFT){double[]vector=new double[rows];for(int c=0;c<columns;c++){for(int r=0;r<rows;r++)vector[r]=alpha*right[r*columns+c];solveTriangular(triangle,transpose,diagonal,rows,matrix,vector);for(int r=0;r<rows;r++)right[r*columns+c]=vector[r];}}else{double[]vector=new double[columns];MatrixTranspose reversed=transpose==MatrixTranspose.NONE?MatrixTranspose.TRANSPOSE:MatrixTranspose.NONE;for(int r=0;r<rows;r++){for(int c=0;c<columns;c++)vector[c]=alpha*right[r*columns+c];solveTriangular(triangle,reversed,diagonal,columns,matrix,vector);System.arraycopy(vector,0,right,r*columns,columns);}}}
	private static void solveTriangular(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int n,double[]a,double[]x){boolean lower=(triangle==MatrixTriangle.LOWER)==(transpose==MatrixTranspose.NONE);if(lower){for(int i=0;i<n;i++){double value=x[i];for(int j=0;j<i;j++)value-=(transpose==MatrixTranspose.NONE?a[i*n+j]:a[j*n+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?value:value/a[i*n+i];}}else{for(int i=n-1;i>=0;i--){double value=x[i];for(int j=i+1;j<n;j++)value-=(transpose==MatrixTranspose.NONE?a[i*n+j]:a[j*n+i])*x[j];x[i]=diagonal==MatrixDiagonal.UNIT?value:value/a[i*n+i];}}}
	private static void checkTriangular(MatrixTriangle triangle,MatrixTranspose transpose,MatrixDiagonal diagonal,int n,double[]a,double[]x){if(triangle==null||transpose==null||diagonal==null||n<1||a==null||a.length!=n*n||x==null||x.length!=n)throw new IllegalArgumentException("invalid triangular solve dimensions");}

	static void dcsrmv(double alpha, CsrMatrix matrix, double[] x, double beta, double[] y) {
		if (matrix == null || x == null || x.length != matrix.columns()
				|| y == null || y.length != matrix.rows())
			throw new IllegalArgumentException("CSR matrix-vector dimensions do not conform");
		double[] values = matrix.values(); int[] columns = matrix.columnIndices(), starts = matrix.rowStarts();
		for (int row = 0; row < matrix.rows(); row++) {
			double sum = 0.0;
			for (int offset = starts[row] - 1; offset < starts[row + 1] - 1; offset++)
				sum += values[offset] * x[columns[offset] - 1];
			y[row] = alpha * sum + beta * y[row];
		}
	}

	static void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		if (matrix == null || rightColumns < 1 || right == null
				|| right.length != matrix.columns() * rightColumns || result == null
				|| result.length != matrix.rows() * rightColumns)
			throw new IllegalArgumentException("CSR matrix-matrix dimensions do not conform");
		double[] values = matrix.values(); int[] columns = matrix.columnIndices(), starts = matrix.rowStarts();
		for (int row = 0; row < matrix.rows(); row++) for (int column = 0; column < rightColumns; column++) {
			double sum = 0.0;
			for (int offset = starts[row] - 1; offset < starts[row + 1] - 1; offset++)
				sum += values[offset] * right[(columns[offset] - 1) * rightColumns + column];
			int index = row * rightColumns + column;
			result[index] = alpha * sum + beta * result[index];
		}
	}

	static CholeskyFactor dpotrf(double[] matrix, int dimension) {
		checkMatrix(dimension, dimension, matrix); checkFinite(matrix);
		double[] lower = new double[matrix.length];
		for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
			double sum = matrix[row * dimension + column];
			for (int k = 0; k < column; k++) sum -= lower[row * dimension + k] * lower[column * dimension + k];
			if (row == column) {
				if (!(sum > 0.0) || !Double.isFinite(sum))
					throw new IllegalArgumentException("matrix is not finite symmetric positive definite at minor " + (row + 1));
				lower[row * dimension + column] = Math.sqrt(sum);
			} else lower[row * dimension + column] = sum / lower[column * dimension + column];
		}
		return new CholeskyFactor(dimension, lower);
	}

	static PivotedQrFactor dgeqp3(double[] matrix, int rows, int columns) {
		checkMatrix(rows, columns, matrix); checkFinite(matrix); double[] qr = matrix.clone();
		int reflectors = Math.min(rows, columns); double[] tau = new double[reflectors];
		int[] pivot = new int[columns]; for (int i = 0; i < columns; i++) pivot[i] = i;
		for (int k = 0; k < reflectors; k++) {
			int selected = k; double selectedNorm = -1.0;
			for (int column = k; column < columns; column++) {
				double norm = 0.0; for (int row = k; row < rows; row++) {
					double value = qr[row * columns + column]; norm += value * value;
				}
				if (norm > selectedNorm) { selectedNorm = norm; selected = column; }
			}
			if (selected != k) {
				for (int row = 0; row < rows; row++) {
					int first = row * columns + k, second = row * columns + selected;
					double value = qr[first]; qr[first] = qr[second]; qr[second] = value;
				}
				int value = pivot[k]; pivot[k] = pivot[selected]; pivot[selected] = value;
			}
			double norm = dnrm2(rows - k, qr, k * columns + k, columns);
			if (norm == 0.0) { tau[k] = 0.0; continue; }
			double alpha = qr[k * columns + k], diagonal = -Math.copySign(norm, alpha);
			tau[k] = (diagonal - alpha) / diagonal;
			double denominator = alpha - diagonal; qr[k * columns + k] = diagonal;
			for (int row = k + 1; row < rows; row++) qr[row * columns + k] /= denominator;
			for (int column = k + 1; column < columns; column++) {
				double product = qr[k * columns + column];
				for (int row = k + 1; row < rows; row++)
					product += qr[row * columns + k] * qr[row * columns + column];
				product *= tau[k]; qr[k * columns + column] -= product;
				for (int row = k + 1; row < rows; row++)
					qr[row * columns + column] -= qr[row * columns + k] * product;
			}
		}
		return new PivotedQrFactor(rows, columns, qr, tau, pivot);
	}

	static SymmetricEigenDecomposition dsyev(double[] matrix, int dimension) {
		checkMatrix(dimension, dimension, matrix); checkFinite(matrix);
		checkSymmetric(matrix, dimension); double[] values = matrix.clone();
		double[] vectors = identity(dimension); double epsilon = 16.0 * Math.ulp(1.0);
		int maximumSweeps = Math.max(32, 8 * dimension);
		for (int sweep = 0; sweep < maximumSweeps; sweep++) {
			boolean changed = false;
			for (int p = 0; p < dimension - 1; p++) for (int q = p + 1; q < dimension; q++) {
				double apq = values[p * dimension + q];
				double threshold = epsilon * (Math.abs(values[p * dimension + p])
						+ Math.abs(values[q * dimension + q]) + 1.0);
				if (Math.abs(apq) <= threshold) continue;
				changed = true; double app = values[p * dimension + p], aqq = values[q * dimension + q];
				double tau = (aqq - app) / (2.0 * apq);
				double t = Math.copySign(1.0 / (Math.abs(tau) + Math.hypot(1.0, tau)), tau);
				double c = 1.0 / Math.hypot(1.0, t), s = t * c;
				for (int k = 0; k < dimension; k++) if (k != p && k != q) {
					double akp = values[k * dimension + p], akq = values[k * dimension + q];
					double nextP = c * akp - s * akq, nextQ = s * akp + c * akq;
					values[k * dimension + p] = values[p * dimension + k] = nextP;
					values[k * dimension + q] = values[q * dimension + k] = nextQ;
				}
				values[p * dimension + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
				values[q * dimension + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
				values[p * dimension + q] = values[q * dimension + p] = 0.0;
				for (int row = 0; row < dimension; row++) {
					double vp = vectors[row * dimension + p], vq = vectors[row * dimension + q];
					vectors[row * dimension + p] = c * vp - s * vq;
					vectors[row * dimension + q] = s * vp + c * vq;
				}
			}
			if (!changed) break;
			if (sweep + 1 == maximumSweeps)
				throw new IllegalStateException("symmetric eigendecomposition did not converge");
		}
		double[] eigenvalues = new double[dimension];
		for (int i = 0; i < dimension; i++) eigenvalues[i] = values[i * dimension + i];
		sortEigenpairs(eigenvalues, vectors, dimension); canonicalizeColumns(vectors, dimension, dimension);
		return new SymmetricEigenDecomposition(dimension, eigenvalues, vectors);
	}

	static SingularValueDecomposition dgesvd(double[] matrix, int rows, int columns) {
		checkMatrix(rows, columns, matrix); checkFinite(matrix);
		if (rows >= columns) return tallSvd(matrix, rows, columns);
		double[] transposed = new double[matrix.length];
		for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
			transposed[column * rows + row] = matrix[row * columns + column];
		SingularValueDecomposition reverse = tallSvd(transposed, columns, rows);
		double[] reverseU = reverse.leftSingularVectors();
		double[] reverseVt = reverse.rightSingularVectorsTransposed();
		double[] left = new double[rows * rows], rightTransposed = new double[rows * columns];
		for (int row = 0; row < rows; row++) for (int column = 0; column < rows; column++)
			left[row * rows + column] = reverseVt[column * rows + row];
		for (int component = 0; component < rows; component++) for (int column = 0; column < columns; column++)
			rightTransposed[component * columns + column] = reverseU[column * rows + component];
		return new SingularValueDecomposition(rows, columns, reverse.singularValues(), left, rightTransposed);
	}

	private static SingularValueDecomposition tallSvd(double[] matrix, int rows, int columns) {
		double[] work = matrix.clone(), vectors = identity(columns);
		double epsilon = 16.0 * Math.ulp(1.0); int maximumSweeps = Math.max(48, 12 * columns);
		for (int sweep = 0; sweep < maximumSweeps; sweep++) {
			boolean changed = false;
			for (int p = 0; p < columns - 1; p++) for (int q = p + 1; q < columns; q++) {
				double alpha = 0.0, beta = 0.0, gamma = 0.0;
				for (int row = 0; row < rows; row++) {
					double left = work[row * columns + p], right = work[row * columns + q];
					alpha += left * left; beta += right * right; gamma += left * right;
				}
				if (gamma == 0.0 || Math.abs(gamma) <= epsilon * Math.sqrt(alpha * beta)) continue;
				changed = true; double zeta = (beta - alpha) / (2.0 * gamma);
				double t = Math.copySign(1.0 / (Math.abs(zeta) + Math.hypot(1.0, zeta)), zeta);
				double c = 1.0 / Math.hypot(1.0, t), s = c * t;
				for (int row = 0; row < rows; row++) {
					int pi = row * columns + p, qi = row * columns + q;
					double left = work[pi], right = work[qi]; work[pi] = c * left - s * right; work[qi] = s * left + c * right;
				}
				for (int row = 0; row < columns; row++) {
					int pi = row * columns + p, qi = row * columns + q;
					double left = vectors[pi], right = vectors[qi]; vectors[pi] = c * left - s * right; vectors[qi] = s * left + c * right;
				}
			}
			if (!changed) break;
			if (sweep + 1 == maximumSweeps) throw new IllegalStateException("SVD did not converge");
		}
		double[] singular = new double[columns];
		for (int column = 0; column < columns; column++) singular[column] = columnNorm(work, rows, columns, column);
		for (int i = 0; i < columns - 1; i++) {
			int selected = i; for (int j = i + 1; j < columns; j++) if (singular[j] > singular[selected]) selected = j;
			if (selected != i) { double value = singular[i]; singular[i] = singular[selected]; singular[selected] = value;
				swapColumns(work, rows, columns, i, selected); swapColumns(vectors, columns, columns, i, selected); }
		}
		double[] left = new double[rows * columns];
		double threshold = Math.max(rows, columns) * Math.ulp(1.0) * singular[0];
		for (int column = 0; column < columns; column++) {
			if (singular[column] > threshold) for (int row = 0; row < rows; row++)
				left[row * columns + column] = work[row * columns + column] / singular[column];
			else completeOrthonormalColumn(left, rows, columns, column);
		}
		canonicalizePairedColumns(left, vectors, rows, columns);
		double[] rightTransposed = new double[columns * columns];
		for (int row = 0; row < columns; row++) for (int column = 0; column < columns; column++)
			rightTransposed[row * columns + column] = vectors[column * columns + row];
		return new SingularValueDecomposition(rows, columns, singular, left, rightTransposed);
	}

	private static double[] identity(int dimension) {
		double[] result = new double[dimension * dimension];
		for (int i = 0; i < dimension; i++) result[i * dimension + i] = 1.0; return result;
	}
	private static double columnNorm(double[] matrix, int rows, int columns, int column) {
		double scale = 0.0, sum = 1.0;
		for (int row = 0; row < rows; row++) { double value = Math.abs(matrix[row * columns + column]);
			if (value != 0.0) { if (scale < value) { double ratio = scale / value; sum = 1.0 + sum * ratio * ratio; scale = value; }
			else { double ratio = value / scale; sum += ratio * ratio; } } }
		return scale == 0.0 ? 0.0 : scale * Math.sqrt(sum);
	}
	private static void swapColumns(double[] matrix, int rows, int columns, int first, int second) {
		for (int row = 0; row < rows; row++) { int a = row * columns + first, b = row * columns + second;
			double value = matrix[a]; matrix[a] = matrix[b]; matrix[b] = value; }
	}
	private static void completeOrthonormalColumn(double[] matrix, int rows, int columns, int column) {
		for (int candidate = 0; candidate < rows; candidate++) {
			for (int row = 0; row < rows; row++) matrix[row * columns + column] = row == candidate ? 1.0 : 0.0;
			for (int previous = 0; previous < column; previous++) { double product = 0.0;
				for (int row = 0; row < rows; row++) product += matrix[row * columns + previous] * matrix[row * columns + column];
				for (int row = 0; row < rows; row++) matrix[row * columns + column] -= product * matrix[row * columns + previous]; }
			double norm = columnNorm(matrix, rows, columns, column);
			if (norm > 16.0 * Math.ulp(1.0)) { for (int row = 0; row < rows; row++) matrix[row * columns + column] /= norm; return; }
		}
		throw new IllegalStateException("cannot complete singular-vector basis");
	}
	private static void sortEigenpairs(double[] values, double[] vectors, int dimension) {
		for (int i = 0; i < dimension - 1; i++) { int selected = i;
			for (int j = i + 1; j < dimension; j++) if (values[j] < values[selected]) selected = j;
			if (selected != i) { double value = values[i]; values[i] = values[selected]; values[selected] = value;
				swapColumns(vectors, dimension, dimension, i, selected); } }
	}
	private static void canonicalizeColumns(double[] matrix, int rows, int columns) {
		for (int column = 0; column < columns; column++) { int largest = 0;
			for (int row = 1; row < rows; row++) if (Math.abs(matrix[row * columns + column]) > Math.abs(matrix[largest * columns + column])) largest = row;
			if (matrix[largest * columns + column] < 0.0) for (int row = 0; row < rows; row++) matrix[row * columns + column] = -matrix[row * columns + column]; }
	}
	private static void canonicalizePairedColumns(double[] left, double[] right, int rows, int columns) {
		for (int column = 0; column < columns; column++) { int largest = 0;
			for (int row = 1; row < rows; row++) if (Math.abs(left[row * columns + column]) > Math.abs(left[largest * columns + column])) largest = row;
			if (left[largest * columns + column] < 0.0) { for (int row = 0; row < rows; row++) left[row * columns + column] = -left[row * columns + column];
				for (int row = 0; row < columns; row++) right[row * columns + column] = -right[row * columns + column]; } }
	}
	private static void checkSymmetric(double[] matrix, int dimension) {
		double scale = 1.0; for (double value : matrix) scale = Math.max(scale, Math.abs(value));
		double tolerance = 64.0 * Math.ulp(1.0) * scale;
		for (int row = 0; row < dimension; row++) for (int column = row + 1; column < dimension; column++)
			if (Math.abs(matrix[row * dimension + column] - matrix[column * dimension + row]) > tolerance)
				throw new IllegalArgumentException("eigenvalue matrix must be symmetric");
	}

	private static void checkRegion(int count, double[] values, int offset, int stride) {
		if (count < 0 || values == null || offset < 0 || stride < 1
				|| (count > 0 && (long) offset + (long) (count - 1) * stride >= values.length))
			throw new IllegalArgumentException("invalid strided vector region");
	}
	private static void checkMatrix(int rows, int columns, double[] matrix) {
		if (rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns)
			throw new IllegalArgumentException("invalid row-major matrix dimensions");
	}
	private static void requireTranspose(MatrixTranspose transpose) {
		if (transpose == null) throw new IllegalArgumentException("matrix transpose policy is required");
	}
	private static void checkFinite(double[] values) {
		for (double value : values) if (!Double.isFinite(value))
			throw new IllegalArgumentException("factorization matrix must be finite");
	}
}
