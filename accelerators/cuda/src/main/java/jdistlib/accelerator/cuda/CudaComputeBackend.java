/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.cuda;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuDeviceTotalMem;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLogSize;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTXSize;

import java.nio.charset.StandardCharsets;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeCapabilities;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.PreparedLogisticRegression;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.UnaryOperation;

/** Optional double-precision CUDA backend using JCuda Driver and JNvrtc. */
public final class CudaComputeBackend implements ComputeBackend {
	private static final int BLOCK = 256;
	private static final String SOURCE =
			"extern \"C\" __global__ void unary_kernel(int op,const double*x,double*y,int n){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){double v=x[i];"
			+ "if(op==0)y[i]=exp(v);else if(op==1)y[i]=log(v);else if(op==2)y[i]=log1p(v);"
			+ "else if(op==3)y[i]=sqrt(v);else if(op==4)y[i]=tanh(v);"
			+ "else y[i]=v>=0?1.0/(1.0+exp(-v)):exp(v)/(1.0+exp(v));}}"
			+ "extern \"C\" __global__ void axpy_kernel(double a,const double*x,const double*y,double*z,int n){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)z[i]=a*x[i]+y[i];}"
			+ "extern \"C\" __global__ void dot_kernel(const double*x,const double*y,double*out,int n){"
			+ "__shared__ double s[256];double v=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)v+=x[i]*y[i];"
			+ "s[threadIdx.x]=v;__syncthreads();for(int w=128;w;w>>=1){if(threadIdx.x<w)s[threadIdx.x]+=s[threadIdx.x+w];__syncthreads();}"
			+ "if(threadIdx.x==0)out[0]=s[0];}"
			+ "extern \"C\" __global__ void gemm_kernel(const double*a,const double*b,double*c,int m,int k,int n){"
			+ "int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;"
			+ "if(row<m&&col<n){double sum=0;for(int p=0;p<k;p++)sum+=a[row*k+p]*b[p*n+col];c[row*n+col]=sum;}}"
			+ "extern \"C\" __global__ void transpose_product(const double*x,const double*v,double*out,int rows,int cols,int batches){"
			+ "int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<cols*batches){int b=z/cols,col=z-b*cols;double sum=0;"
			+ "for(int row=0;row<rows;row++)sum+=x[row*cols+col]*v[b*rows+row];out[z]=sum;}}"
			+ "__device__ double l1e(double x){return x>0?x+log1p(exp(-x)):log1p(exp(x));}"
			+ "extern \"C\" __global__ void logistic_residual(const double*x,const double*y,const double*q,double*r,double*t,int rows,int dims,int chains){"
			+ "int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<rows*chains){int c=z/rows,i=z-c*rows;double eta=0;"
			+ "for(int d=0;d<dims;d++)eta+=x[i*dims+d]*q[c*dims+d];double p=eta>=0?1.0/(1.0+exp(-eta)):exp(eta)/(1.0+exp(eta));"
			+ "r[z]=y[i]-p;t[z]=y[i]*eta-l1e(eta);}}"
			+ "extern \"C\" __global__ void logistic_gradient(const double*x,const double*q,const double*r,double*g,int rows,int dims,int chains,double prior){"
			+ "int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<dims*chains){int c=z/dims,d=z-c*dims;double sum=-prior*q[z];"
			+ "for(int i=0;i<rows;i++)sum+=r[c*rows+i]*x[i*dims+d];g[z]=sum;}}"
			+ "extern \"C\" __global__ void logistic_logp(const double*t,const double*q,double*out,int rows,int dims,double prior){"
			+ "__shared__ double s[256];int c=blockIdx.x;double sum=0;for(int i=threadIdx.x;i<rows;i+=blockDim.x)sum+=t[c*rows+i];"
			+ "for(int d=threadIdx.x;d<dims;d+=blockDim.x){double v=q[c*dims+d];sum-=0.5*prior*v*v;}s[threadIdx.x]=sum;__syncthreads();"
			+ "for(int w=128;w;w>>=1){if(threadIdx.x<w)s[threadIdx.x]+=s[threadIdx.x+w];__syncthreads();}if(threadIdx.x==0)out[c]=s[0];}";

	private CUcontext context;
	private CUmodule module;
	private ComputeCapabilities capabilities;
	private Throwable unavailableCause;

	/** Detects device zero and makes this instance unavailable if CUDA or NVRTC cannot initialize. */
	public CudaComputeBackend() {
		try { initialize(); } catch (Throwable error) { unavailableCause = error; close(); }
	}
	@Override public String id() { return "cuda"; }
	@Override public boolean available() { return unavailableCause == null && context != null; }
	@Override public ComputeCapabilities capabilities() { ensureAvailable(); return capabilities; }
	/** Reports why optional CUDA initialization failed.
	 * @return the initialization failure, or {@code null} when available
	 */
	public Throwable unavailableCause() { return unavailableCause; }

	@Override public synchronized double[] unary(UnaryOperation operation, double[] input) {
		ensureAvailable(); if (operation == null || input == null) throw new IllegalArgumentException("operation and input required");
		setCurrent(); CUdeviceptr x = allocate(input), y = allocateDoubles(input.length);
		try {
			launch("unary_kernel", grid(input.length), 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(new int[] {operation.ordinal()}), Pointer.to(x),
							Pointer.to(y), Pointer.to(new int[] {input.length})));
			return copy(y, input.length);
		} finally { cuMemFree(x); cuMemFree(y); }
	}
	@Override public synchronized double[] axpy(double alpha, double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); setCurrent();
		CUdeviceptr dx = allocate(x), dy = allocate(y), result = allocateDoubles(x.length);
		try {
			launch("axpy_kernel", grid(x.length), 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(new double[] {alpha}), Pointer.to(dx), Pointer.to(dy),
							Pointer.to(result), Pointer.to(new int[] {x.length})));
			return copy(result, x.length);
		} finally { cuMemFree(dx); cuMemFree(dy); cuMemFree(result); }
	}
	@Override public synchronized double dot(double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); setCurrent();
		CUdeviceptr dx = allocate(x), dy = allocate(y), result = allocateDoubles(1);
		try {
			launch("dot_kernel", 1, 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(dx), Pointer.to(dy), Pointer.to(result),
							Pointer.to(new int[] {x.length})));
			return copy(result, 1)[0];
		} finally { cuMemFree(dx); cuMemFree(dy); cuMemFree(result); }
	}
	@Override public synchronized double[][] matrixMultiply(double[][] left, double[][] right) {
		int[] aShape = shape(left), bShape = shape(right);
		if (aShape[1] != bShape[0]) throw new IllegalArgumentException("matrix dimensions do not conform");
		ensureAvailable(); setCurrent(); double[] a = flatten(left), b = flatten(right);
		CUdeviceptr da = allocate(a), db = allocate(b), dc = allocateDoubles(aShape[0] * bShape[1]);
		try {
			launch("gemm_kernel", (bShape[1] + 15) / 16, (aShape[0] + 15) / 16, 1,
					16, 16, 1, Pointer.to(Pointer.to(da), Pointer.to(db), Pointer.to(dc),
							Pointer.to(new int[] {aShape[0]}), Pointer.to(new int[] {aShape[1]}),
							Pointer.to(new int[] {bShape[1]})));
			return reshape(copy(dc, aShape[0] * bShape[1]), aShape[0], bShape[1]);
		} finally { cuMemFree(da); cuMemFree(db); cuMemFree(dc); }
	}
	@Override public synchronized LogisticRegressionBatchResult logisticRegression(
			double[][] design, double[] outcomes, double[][] states, double priorPrecision) {
		int[] xShape = shape(design), qShape = shape(states);
		if (outcomes == null || outcomes.length != xShape[0] || qShape[1] != xShape[1]
				|| !(priorPrecision >= 0.0)) throw new IllegalArgumentException("invalid logistic batch");
		ensureAvailable(); setCurrent(); int rows = xShape[0], dims = xShape[1], chains = qShape[0];
		CUdeviceptr x = allocate(flatten(design)), y = allocate(outcomes), q = allocate(flatten(states));
		CUdeviceptr residual = allocateDoubles(rows * chains), terms = allocateDoubles(rows * chains);
		CUdeviceptr gradient = allocateDoubles(dims * chains), logp = allocateDoubles(chains);
		try {
			launch("logistic_residual", grid(rows * chains), 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(x), Pointer.to(y), Pointer.to(q), Pointer.to(residual),
							Pointer.to(terms), Pointer.to(new int[] {rows}), Pointer.to(new int[] {dims}),
							Pointer.to(new int[] {chains})));
			launch("logistic_gradient", grid(dims * chains), 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(x), Pointer.to(q), Pointer.to(residual), Pointer.to(gradient),
							Pointer.to(new int[] {rows}), Pointer.to(new int[] {dims}),
							Pointer.to(new int[] {chains}), Pointer.to(new double[] {priorPrecision})));
			launch("logistic_logp", chains, 1, 1, BLOCK, 1, 1,
					Pointer.to(Pointer.to(terms), Pointer.to(q), Pointer.to(logp),
							Pointer.to(new int[] {rows}), Pointer.to(new int[] {dims}),
							Pointer.to(new double[] {priorPrecision})));
			return new LogisticRegressionBatchResult(copy(logp, chains),
					reshape(copy(gradient, dims * chains), chains, dims));
		} finally {
			cuMemFree(x); cuMemFree(y); cuMemFree(q); cuMemFree(residual);
			cuMemFree(terms); cuMemFree(gradient); cuMemFree(logp);
		}
	}
	@Override public synchronized PreparedTransposeProduct prepareTransposeProduct(double[][] matrix) {
		int[] matrixShape = shape(matrix); ensureAvailable(); setCurrent();
		return new PreparedTranspose(matrix, matrixShape[0], matrixShape[1]);
	}

	private final class PreparedTranspose implements PreparedTransposeProduct {
		private final int rows, columns; private CUdeviceptr matrix;
		PreparedTranspose(double[][] source, int rows, int columns) {
			this.rows = rows; this.columns = columns; matrix = allocate(flatten(source));
		}
		@Override public int rows() { return rows; }
		@Override public int columns() { return columns; }
		@Override public double[][] multiply(double[][] vectors) {
			synchronized (CudaComputeBackend.this) {
				if (matrix == null) throw new IllegalStateException("prepared transpose product is closed");
				int[] vectorShape = shape(vectors); if (vectorShape[1] != rows) throw new IllegalArgumentException("score vector length mismatch");
				ensureAvailable(); setCurrent(); int batches = vectorShape[0], count = columns * batches;
				CUdeviceptr input = allocate(flatten(vectors)), output = allocateDoubles(count);
				try {
					launch("transpose_product", grid(count), 1, 1, BLOCK, 1, 1,
							Pointer.to(Pointer.to(matrix), Pointer.to(input), Pointer.to(output),
									Pointer.to(new int[] {rows}), Pointer.to(new int[] {columns}), Pointer.to(new int[] {batches})));
					return reshape(copy(output, count), batches, columns);
				} finally { cuMemFree(input); cuMemFree(output); }
			}
		}
		@Override public void close() { synchronized (CudaComputeBackend.this) {
			if (matrix != null) { setCurrent(); cuMemFree(matrix); matrix = null; }
		} }
	}
	@Override public synchronized PreparedLogisticRegression prepareLogisticRegression(
			double[][] design, double[] outcomes) {
		int[] xShape = shape(design);
		if (outcomes == null || outcomes.length != xShape[0])
			throw new IllegalArgumentException("one outcome per row is required");
		ensureAvailable(); setCurrent();
		return new PreparedLogistic(design, outcomes, xShape[0], xShape[1]);
	}

	private final class PreparedLogistic implements PreparedLogisticRegression {
		private final int rows, dimensions;
		private CUdeviceptr design, outcomes;
		PreparedLogistic(double[][] sourceDesign, double[] sourceOutcomes, int rows,
				int dimensions) {
			this.rows = rows; this.dimensions = dimensions;
			design = allocate(flatten(sourceDesign)); outcomes = allocate(sourceOutcomes);
		}
		@Override public int rows() { return rows; }
		@Override public int dimensions() { return dimensions; }
		@Override public LogisticRegressionBatchResult evaluate(double[][] states,
				double priorPrecision) {
			synchronized (CudaComputeBackend.this) {
				if (design == null) throw new IllegalStateException("prepared likelihood is closed");
				int[] qShape = shape(states);
				if (qShape[1] != dimensions || !(priorPrecision >= 0.0))
					throw new IllegalArgumentException("invalid logistic batch");
				ensureAvailable(); setCurrent(); int chains = qShape[0];
				CUdeviceptr q = allocate(flatten(states));
				CUdeviceptr residual = allocateDoubles(rows * chains), terms = allocateDoubles(rows * chains);
				CUdeviceptr gradient = allocateDoubles(dimensions * chains), logp = allocateDoubles(chains);
				try {
					launch("logistic_residual", grid(rows * chains), 1, 1, BLOCK, 1, 1,
							Pointer.to(Pointer.to(design), Pointer.to(outcomes), Pointer.to(q),
									Pointer.to(residual), Pointer.to(terms), Pointer.to(new int[] {rows}),
									Pointer.to(new int[] {dimensions}), Pointer.to(new int[] {chains})));
					launch("logistic_gradient", grid(dimensions * chains), 1, 1, BLOCK, 1, 1,
							Pointer.to(Pointer.to(design), Pointer.to(q), Pointer.to(residual),
									Pointer.to(gradient), Pointer.to(new int[] {rows}),
									Pointer.to(new int[] {dimensions}), Pointer.to(new int[] {chains}),
									Pointer.to(new double[] {priorPrecision})));
					launch("logistic_logp", chains, 1, 1, BLOCK, 1, 1,
							Pointer.to(Pointer.to(terms), Pointer.to(q), Pointer.to(logp),
									Pointer.to(new int[] {rows}), Pointer.to(new int[] {dimensions}),
									Pointer.to(new double[] {priorPrecision})));
					return new LogisticRegressionBatchResult(copy(logp, chains),
							reshape(copy(gradient, dimensions * chains), chains, dimensions));
				} finally {
					cuMemFree(q); cuMemFree(residual); cuMemFree(terms);
					cuMemFree(gradient); cuMemFree(logp);
				}
			}
		}
		@Override public void close() {
			synchronized (CudaComputeBackend.this) {
				if (design != null) { setCurrent(); cuMemFree(design); cuMemFree(outcomes); design = null; outcomes = null; }
			}
		}
	}

	private void initialize() {
		JCudaDriver.setExceptionsEnabled(true); JNvrtc.setExceptionsEnabled(true); cuInit(0);
		CUdevice device = new CUdevice(); cuDeviceGet(device, 0);
		context = new CUcontext(); cuCtxCreate(context, 0, device);
		byte[] nameBytes = new byte[256]; cuDeviceGetName(nameBytes, nameBytes.length, device);
		int end = 0; while (end < nameBytes.length && nameBytes[end] != 0) end++;
		String name = new String(nameBytes, 0, end, StandardCharsets.UTF_8);
		long[] memory = new long[1]; cuDeviceTotalMem(memory, device);
		int[] major = new int[1], minor = new int[1];
		cuDeviceGetAttribute(major, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		cuDeviceGetAttribute(minor, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
		capabilities = new ComputeCapabilities("CUDA", name + " (sm_" + major[0] + minor[0] + ")",
				major[0] >= 2, true, memory[0]);
		String ptx = compile(SOURCE, "--std=c++11", "--gpu-architecture=compute_" + major[0] + minor[0]);
		module = new CUmodule(); cuModuleLoadData(module, ptx);
	}
	private static String compile(String source, String... options) {
		nvrtcProgram program = new nvrtcProgram(); nvrtcCreateProgram(program, source, "jdistlib.cu", 0, null, null);
		try {
			nvrtcCompileProgram(program, options.length, options);
			String[] result = new String[1]; nvrtcGetPTX(program, result); return result[0];
		} catch (RuntimeException error) {
			String[] log = new String[1]; nvrtcGetProgramLog(program, log);
			throw new IllegalStateException(log[0], error);
		} finally { nvrtcDestroyProgram(program); }
	}
	private void launch(String name, int gx, int gy, int gz, int bx, int by, int bz, Pointer parameters) {
		CUfunction function = new CUfunction(); cuModuleGetFunction(function, module, name);
		cuLaunchKernel(function, gx, gy, gz, bx, by, bz, 0, null, parameters, null);
	}
	private static CUdeviceptr allocate(double[] values) {
		CUdeviceptr result = allocateDoubles(values.length);
		cuMemcpyHtoD(result, Pointer.to(values), (long) values.length * Sizeof.DOUBLE); return result;
	}
	private static CUdeviceptr allocateDoubles(int count) {
		CUdeviceptr result = new CUdeviceptr(); cuMemAlloc(result, (long) count * Sizeof.DOUBLE); return result;
	}
	private static double[] copy(CUdeviceptr source, int count) {
		double[] result = new double[count]; cuMemcpyDtoH(Pointer.to(result), source, (long) count * Sizeof.DOUBLE); return result;
	}
	private static int grid(int count) { return (count + BLOCK - 1) / BLOCK; }
	private void setCurrent() { cuCtxSetCurrent(context); }
	private void ensureAvailable() {
		if (!available()) throw new IllegalStateException("CUDA backend unavailable", unavailableCause);
	}
	private static void checkVectors(double[] x, double[] y) {
		if (x == null || y == null || x.length != y.length) throw new IllegalArgumentException("vector lengths must match");
	}
	private static int[] shape(double[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			throw new IllegalArgumentException("matrix must be nonempty");
		int columns = matrix[0].length;
		for (double[] row : matrix) if (row == null || row.length != columns)
			throw new IllegalArgumentException("matrix must be rectangular");
		return new int[] {matrix.length, columns};
	}
	private static double[] flatten(double[][] matrix) {
		int[] shape = shape(matrix); double[] result = new double[shape[0] * shape[1]];
		for (int i = 0; i < shape[0]; i++) System.arraycopy(matrix[i], 0, result, i * shape[1], shape[1]);
		return result;
	}
	private static double[][] reshape(double[] values, int rows, int columns) {
		double[][] result = new double[rows][columns];
		for (int i = 0; i < rows; i++) System.arraycopy(values, i * columns, result[i], 0, columns);
		return result;
	}
	@Override public synchronized void close() {
		try { if (context != null) cuCtxSetCurrent(context); } catch (Throwable ignored) { /* optional backend */ }
		try { if (module != null) cuModuleUnload(module); } catch (Throwable ignored) { /* optional backend */ }
		try { if (context != null) cuCtxDestroy(context); } catch (Throwable ignored) { /* optional backend */ }
		module = null; context = null;
	}
}
