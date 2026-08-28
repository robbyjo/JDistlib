/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.opencl;

import static org.jocl.CL.*;

import java.nio.charset.StandardCharsets;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import jdistlib.accelerator.ComputeBackend;
import jdistlib.accelerator.ComputeCapabilities;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.PreparedLogisticRegression;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.UnaryOperation;

/** Optional portable OpenCL 1.2 backend using JOCL. */
public final class OpenClComputeBackend implements ComputeBackend {
	private static final long LOCAL = 256L;
	private static final String SOURCE =
			"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
			+ "double logistic(double v){return v>=0?1.0/(1.0+exp(-v)):exp(v)/(1.0+exp(v));}\n"
			+ "double l1e(double v){return v>0?v+log1p(exp(-v)):log1p(exp(v));}\n"
			+ "__kernel void unary_kernel(int op,__global const double*x,__global double*y,int n){int i=get_global_id(0);if(i<n){double v=x[i];if(op==0)y[i]=exp(v);else if(op==1)y[i]=log(v);else if(op==2)y[i]=log1p(v);else if(op==3)y[i]=sqrt(v);else if(op==4)y[i]=tanh(v);else y[i]=logistic(v);}}\n"
			+ "__kernel void axpy_kernel(double a,__global const double*x,__global const double*y,__global double*z,int n){int i=get_global_id(0);if(i<n)z[i]=a*x[i]+y[i];}\n"
			+ "__kernel void dot_kernel(__global const double*x,__global const double*y,__global double*out,int n){if(get_global_id(0)==0){double s=0;for(int i=0;i<n;i++)s+=x[i]*y[i];out[0]=s;}}\n"
			+ "__kernel void gemm_kernel(__global const double*a,__global const double*b,__global double*c,int m,int k,int n){int row=get_global_id(1),col=get_global_id(0);if(row<m&&col<n){double s=0;for(int p=0;p<k;p++)s+=a[row*k+p]*b[p*n+col];c[row*n+col]=s;}}\n"
			+ "__kernel void transpose_product(__global const double*x,__global const double*v,__global double*out,int rows,int cols,int batches){int z=get_global_id(0);if(z<cols*batches){int b=z/cols,col=z-b*cols;double s=0;for(int row=0;row<rows;row++)s+=x[row*cols+col]*v[b*rows+row];out[z]=s;}}\n"
			+ "__kernel void logistic_residual(__global const double*x,__global const double*y,__global const double*q,__global double*r,__global double*t,int rows,int dims,int chains){int z=get_global_id(0);if(z<rows*chains){int c=z/rows,i=z-c*rows;double eta=0;for(int d=0;d<dims;d++)eta+=x[i*dims+d]*q[c*dims+d];r[z]=y[i]-logistic(eta);t[z]=y[i]*eta-l1e(eta);}}\n"
			+ "__kernel void logistic_gradient(__global const double*x,__global const double*q,__global const double*r,__global double*g,int rows,int dims,int chains,double prior){int z=get_global_id(0);if(z<dims*chains){int c=z/dims,d=z-c*dims;double s=-prior*q[z];for(int i=0;i<rows;i++)s+=r[c*rows+i]*x[i*dims+d];g[z]=s;}}\n"
			+ "__kernel void logistic_logp(__global const double*t,__global const double*q,__global double*out,int rows,int dims,int chains,double prior){int c=get_global_id(0);if(c<chains){double s=0;for(int i=0;i<rows;i++)s+=t[c*rows+i];for(int d=0;d<dims;d++){double v=q[c*dims+d];s-=0.5*prior*v*v;}out[c]=s;}}\n";

	private cl_context context;
	private cl_command_queue queue;
	private cl_program program;
	private ComputeCapabilities capabilities;
	private Throwable unavailableCause;

	/** Detects the first FP64 GPU and makes this instance unavailable when none can initialize. */
	public OpenClComputeBackend() {
		try { initialize(); } catch (Throwable error) { unavailableCause = error; close(); }
	}
	@Override public String id() { return "opencl"; }
	@Override public boolean available() { return unavailableCause == null && context != null; }
	@Override public ComputeCapabilities capabilities() { ensureAvailable(); return capabilities; }
	/** Reports why optional OpenCL initialization failed.
	 * @return the initialization failure, or {@code null} when available
	 */
	public Throwable unavailableCause() { return unavailableCause; }

	@Override public synchronized double[] unary(UnaryOperation operation, double[] input) {
		if (operation == null || input == null) throw new IllegalArgumentException("operation and input required");
		ensureAvailable(); cl_mem x = input(input), out = output(input.length);
		cl_kernel kernel = kernel("unary_kernel");
		try {
			arg(kernel, 0, operation.ordinal()); arg(kernel, 1, x); arg(kernel, 2, out); arg(kernel, 3, input.length);
			run1d(kernel, input.length); return read(out, input.length);
		} finally { clReleaseKernel(kernel); clReleaseMemObject(x); clReleaseMemObject(out); }
	}
	@Override public synchronized double[] axpy(double alpha, double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); cl_mem dx = input(x), dy = input(y), out = output(x.length);
		cl_kernel kernel = kernel("axpy_kernel");
		try {
			arg(kernel, 0, alpha); arg(kernel, 1, dx); arg(kernel, 2, dy); arg(kernel, 3, out); arg(kernel, 4, x.length);
			run1d(kernel, x.length); return read(out, x.length);
		} finally { clReleaseKernel(kernel); clReleaseMemObject(dx); clReleaseMemObject(dy); clReleaseMemObject(out); }
	}
	@Override public synchronized double dot(double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); cl_mem dx = input(x), dy = input(y), out = output(1);
		cl_kernel kernel = kernel("dot_kernel");
		try {
			arg(kernel, 0, dx); arg(kernel, 1, dy); arg(kernel, 2, out); arg(kernel, 3, x.length);
			run1d(kernel, 1); return read(out, 1)[0];
		} finally { clReleaseKernel(kernel); clReleaseMemObject(dx); clReleaseMemObject(dy); clReleaseMemObject(out); }
	}
	@Override public synchronized double[][] matrixMultiply(double[][] left, double[][] right) {
		int[] aShape = shape(left), bShape = shape(right);
		if (aShape[1] != bShape[0]) throw new IllegalArgumentException("matrix dimensions do not conform");
		ensureAvailable(); cl_mem a = input(flatten(left)), b = input(flatten(right));
		cl_mem out = output(aShape[0] * bShape[1]); cl_kernel kernel = kernel("gemm_kernel");
		try {
			arg(kernel, 0, a); arg(kernel, 1, b); arg(kernel, 2, out); arg(kernel, 3, aShape[0]);
			arg(kernel, 4, aShape[1]); arg(kernel, 5, bShape[1]);
			check(clEnqueueNDRangeKernel(queue, kernel, 2, null,
					new long[] {round(bShape[1]), round(aShape[0])}, new long[] {16, 16}, 0, null, null));
			return reshape(read(out, aShape[0] * bShape[1]), aShape[0], bShape[1]);
		} finally { clReleaseKernel(kernel); clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(out); }
	}
	@Override public synchronized LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		PreparedLogisticRegression prepared = prepareLogisticRegression(design, outcomes);
		try { return prepared.evaluate(states, priorPrecision); } finally { prepared.close(); }
	}
	@Override public synchronized PreparedTransposeProduct prepareTransposeProduct(double[][] matrix) {
		int[] matrixShape = shape(matrix); ensureAvailable();
		return new PreparedTranspose(matrix, matrixShape[0], matrixShape[1]);
	}

	private final class PreparedTranspose implements PreparedTransposeProduct {
		private final int rows, columns; private cl_mem matrix;
		PreparedTranspose(double[][] source, int rows, int columns) {
			this.rows = rows; this.columns = columns; matrix = input(flatten(source));
		}
		@Override public int rows() { return rows; }
		@Override public int columns() { return columns; }
		@Override public double[][] multiply(double[][] vectors) {
			synchronized (OpenClComputeBackend.this) {
				if (matrix == null) throw new IllegalStateException("prepared transpose product is closed");
				int[] vectorShape = shape(vectors); if (vectorShape[1] != rows) throw new IllegalArgumentException("score vector length mismatch");
				int batches = vectorShape[0], count = columns * batches;
				cl_mem input = input(flatten(vectors)), output = output(count); cl_kernel product = kernel("transpose_product");
				try {
					arg(product, 0, matrix); arg(product, 1, input); arg(product, 2, output); arg(product, 3, rows);
					arg(product, 4, columns); arg(product, 5, batches); run1d(product, count);
					return reshape(read(output, count), batches, columns);
				} finally { clReleaseKernel(product); clReleaseMemObject(input); clReleaseMemObject(output); }
			}
		}
		@Override public void close() { synchronized (OpenClComputeBackend.this) {
			if (matrix != null) { clReleaseMemObject(matrix); matrix = null; }
		} }
	}
	@Override public synchronized PreparedLogisticRegression prepareLogisticRegression(double[][] design,
			double[] outcomes) {
		int[] xShape = shape(design);
		if (outcomes == null || outcomes.length != xShape[0])
			throw new IllegalArgumentException("one outcome per row is required");
		ensureAvailable(); return new PreparedLogistic(design, outcomes, xShape[0], xShape[1]);
	}

	private final class PreparedLogistic implements PreparedLogisticRegression {
		private final int rows, dimensions;
		private cl_mem design, outcomes;
		PreparedLogistic(double[][] x, double[] y, int rows, int dimensions) {
			this.rows = rows; this.dimensions = dimensions; design = input(flatten(x)); outcomes = input(y);
		}
		@Override public int rows() { return rows; }
		@Override public int dimensions() { return dimensions; }
		@Override public LogisticRegressionBatchResult evaluate(double[][] states, double priorPrecision) {
			synchronized (OpenClComputeBackend.this) {
				if (design == null) throw new IllegalStateException("prepared likelihood is closed");
				int[] qShape = shape(states); int chains = qShape[0];
				if (qShape[1] != dimensions || !(priorPrecision >= 0.0))
					throw new IllegalArgumentException("invalid logistic batch");
				cl_mem q = input(flatten(states)), residual = output(rows * chains), terms = output(rows * chains);
				cl_mem gradient = output(dimensions * chains), logp = output(chains);
				cl_kernel residualKernel = kernel("logistic_residual"), gradientKernel = kernel("logistic_gradient");
				cl_kernel logpKernel = kernel("logistic_logp");
				try {
					arg(residualKernel, 0, design); arg(residualKernel, 1, outcomes); arg(residualKernel, 2, q);
					arg(residualKernel, 3, residual); arg(residualKernel, 4, terms); arg(residualKernel, 5, rows);
					arg(residualKernel, 6, dimensions); arg(residualKernel, 7, chains); run1d(residualKernel, rows * chains);
					arg(gradientKernel, 0, design); arg(gradientKernel, 1, q); arg(gradientKernel, 2, residual);
					arg(gradientKernel, 3, gradient); arg(gradientKernel, 4, rows); arg(gradientKernel, 5, dimensions);
					arg(gradientKernel, 6, chains); arg(gradientKernel, 7, priorPrecision); run1d(gradientKernel, dimensions * chains);
					arg(logpKernel, 0, terms); arg(logpKernel, 1, q); arg(logpKernel, 2, logp); arg(logpKernel, 3, rows);
					arg(logpKernel, 4, dimensions); arg(logpKernel, 5, chains); arg(logpKernel, 6, priorPrecision); run1d(logpKernel, chains);
					return new LogisticRegressionBatchResult(read(logp, chains),
							reshape(read(gradient, dimensions * chains), chains, dimensions));
				} finally {
					clReleaseKernel(residualKernel); clReleaseKernel(gradientKernel); clReleaseKernel(logpKernel);
					clReleaseMemObject(q); clReleaseMemObject(residual); clReleaseMemObject(terms);
					clReleaseMemObject(gradient); clReleaseMemObject(logp);
				}
			}
		}
		@Override public void close() { synchronized (OpenClComputeBackend.this) {
			if (design != null) { clReleaseMemObject(design); clReleaseMemObject(outcomes); design = null; outcomes = null; }
		} }
	}

	private void initialize() {
		setExceptionsEnabled(false); int[] platformCount = new int[1]; check(clGetPlatformIDs(0, null, platformCount));
		if (platformCount[0] == 0) throw new IllegalStateException("no OpenCL platform");
		cl_platform_id[] platforms = new cl_platform_id[platformCount[0]]; check(clGetPlatformIDs(platforms.length, platforms, null));
		cl_device_id device = null; cl_platform_id platform = null;
		for (cl_platform_id candidate : platforms) {
			int[] count = new int[1]; int status = clGetDeviceIDs(candidate, CL_DEVICE_TYPE_GPU, 0, null, count);
			if (status == CL_SUCCESS && count[0] > 0) { cl_device_id[] devices = new cl_device_id[count[0]];
				check(clGetDeviceIDs(candidate, CL_DEVICE_TYPE_GPU, devices.length, devices, null)); device = devices[0]; platform = candidate; break; }
		}
		if (device == null) throw new IllegalStateException("no OpenCL GPU device");
		cl_context_properties properties = new cl_context_properties(); properties.addProperty(CL_CONTEXT_PLATFORM, platform);
		int[] status = new int[1]; context = clCreateContext(properties, 1, new cl_device_id[] {device}, null, null, status); check(status[0]);
		queue = createQueue(context, device, status); check(status[0]);
		program = clCreateProgramWithSource(context, 1, new String[] {SOURCE}, null, status); check(status[0]);
		int build = clBuildProgram(program, 0, null, null, null, null);
		if (build != CL_SUCCESS) throw new IllegalStateException("OpenCL build failed: " + programBuildLog(program, device));
		long[] memory = new long[1]; check(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(memory), null));
		String extensions = deviceString(device, CL_DEVICE_EXTENSIONS);
		capabilities = new ComputeCapabilities("OpenCL", deviceString(device, CL_DEVICE_NAME),
				extensions.contains("cl_khr_fp64") || extensions.contains("cl_amd_fp64"), true, memory[0]);
		if (!capabilities.doublePrecision()) throw new IllegalStateException("OpenCL device lacks FP64");
	}
	private cl_kernel kernel(String name) { int[] status = new int[1]; cl_kernel result = clCreateKernel(program, name, status); check(status[0]); return result; }
	private cl_mem input(double[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_double, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem output(int count) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE,
			(long) count * Sizeof.cl_double, null, status); check(status[0]); return result; }
	private double[] read(cl_mem source, int count) { double[] result = new double[count]; check(clEnqueueReadBuffer(queue,
			source, CL_TRUE, 0, (long) count * Sizeof.cl_double, Pointer.to(result), 0, null, null)); return result; }
	private void run1d(cl_kernel kernel, int count) { check(clEnqueueNDRangeKernel(queue, kernel, 1, null,
			new long[] {round(count)}, new long[] {LOCAL}, 0, null, null)); }
	private static void arg(cl_kernel kernel, int index, cl_mem value) { check(clSetKernelArg(kernel, index, Sizeof.cl_mem, Pointer.to(value))); }
	private static void arg(cl_kernel kernel, int index, int value) { check(clSetKernelArg(kernel, index, Sizeof.cl_int, Pointer.to(new int[] {value}))); }
	private static void arg(cl_kernel kernel, int index, double value) { check(clSetKernelArg(kernel, index, Sizeof.cl_double, Pointer.to(new double[] {value}))); }
	@SuppressWarnings("deprecation")
	private static cl_command_queue createQueue(cl_context context, cl_device_id device, int[] status) {
		cl_command_queue result = null;
		try { result = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), status); }
		catch (LinkageError unavailableOnOpenCl12) { status[0] = CL_INVALID_OPERATION; }
		if (status[0] != CL_SUCCESS || result == null) result = clCreateCommandQueue(context, device, 0, status);
		return result;
	}
	private static long round(int count) { return ((count + LOCAL - 1L) / LOCAL) * LOCAL; }
	private static String deviceString(cl_device_id device, int parameter) { long[] size = new long[1]; check(clGetDeviceInfo(device, parameter, 0, null, size));
		byte[] bytes = new byte[(int) size[0]]; check(clGetDeviceInfo(device, parameter, bytes.length, Pointer.to(bytes), null));
		int length = bytes.length; while (length > 0 && bytes[length - 1] == 0) length--; return new String(bytes, 0, length, StandardCharsets.UTF_8); }
	private static String programBuildLog(cl_program program, cl_device_id device) { long[] size = new long[1]; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, size);
		byte[] bytes = new byte[(int) size[0]]; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, bytes.length, Pointer.to(bytes), null);
		return new String(bytes, StandardCharsets.UTF_8).trim(); }
	private static void check(int status) { if (status != CL_SUCCESS) throw new IllegalStateException("OpenCL error " + status); }
	private void ensureAvailable() { if (!available()) throw new IllegalStateException("OpenCL unavailable", unavailableCause); }
	@Override public synchronized void close() { if (program != null) { clReleaseProgram(program); program = null; }
		if (queue != null) { clReleaseCommandQueue(queue); queue = null; } if (context != null) { clReleaseContext(context); context = null; } }
	private static void checkVectors(double[] x, double[] y) { if (x == null || y == null || x.length != y.length) throw new IllegalArgumentException("vector lengths must match"); }
	private static int[] shape(double[][] matrix) { if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) throw new IllegalArgumentException("matrix must be nonempty");
		int columns = matrix[0].length; for (double[] row : matrix) if (row == null || row.length != columns) throw new IllegalArgumentException("matrix must be rectangular"); return new int[] {matrix.length, columns}; }
	private static double[] flatten(double[][] matrix) { int[] shape = shape(matrix); double[] result = new double[shape[0] * shape[1]]; int offset = 0;
		for (double[] row : matrix) { System.arraycopy(row, 0, result, offset, row.length); offset += row.length; } return result; }
	private static double[][] reshape(double[] values, int rows, int columns) { double[][] result = new double[rows][columns];
		for (int row = 0; row < rows; row++) System.arraycopy(values, row * columns, result[row], 0, columns); return result; }
}
