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
import jdistlib.accelerator.CholeskyFactor;
import jdistlib.accelerator.FloatCholeskyFactor;
import jdistlib.accelerator.FloatPivotedQrFactor;
import jdistlib.accelerator.FloatSingularValueDecomposition;
import jdistlib.accelerator.FloatSymmetricEigenDecomposition;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.PreparedLogisticRegression;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.PivotedQrFactor;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

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
			+ "extern \"C\" __global__ void blas_axpy(int n,double alpha,const double*x,int xo,int xs,double*y,int yo,int ys){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[yo+i*ys]=alpha*x[xo+i*xs]+y[yo+i*ys];}"
			+ "extern \"C\" __global__ void blas_dot(int n,const double*x,int xo,int xs,const double*y,int yo,int ys,double*out){"
			+ "if(blockIdx.x==0&&threadIdx.x==0){double s=0;for(int i=0;i<n;i++)s+=x[xo+i*xs]*y[yo+i*ys];out[0]=s;}}"
			+ "extern \"C\" __global__ void blas_nrm2(int n,const double*x,int xo,int xs,double*out){"
			+ "if(blockIdx.x==0&&threadIdx.x==0){double scale=0,sum=1;for(int i=0;i<n;i++){double v=fabs(x[xo+i*xs]);"
			+ "if(v!=0){if(scale<v){double r=scale/v;sum=1+sum*r*r;scale=v;}else{double r=v/scale;sum+=r*r;}}}out[0]=scale==0?0:scale*sqrt(sum);}}"
			+ "extern \"C\" __global__ void blas_gemv(int tr,int rows,int cols,double alpha,const double*a,const double*x,double beta,double*y){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x,output=tr?cols:rows,input=tr?rows:cols;if(i<output){double s=0;"
			+ "for(int j=0;j<input;j++)s+=(tr?a[j*cols+i]:a[i*cols+j])*x[j];y[i]=alpha*s+beta*y[i];}}"
			+ "extern \"C\" __global__ void blas_gemm(int ta,int tb,int m,int n,int k,double alpha,const double*a,const double*b,double beta,double*c){"
			+ "int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;if(row<m&&col<n){double s=0;"
			+ "for(int q=0;q<k;q++)s+=(ta?a[q*m+row]:a[row*k+q])*(tb?b[col*k+q]:b[q*n+col]);int z=row*n+col;c[z]=alpha*s+beta*c[z];}}"
			+ "extern \"C\" __global__ void csr_mv(int rows,double alpha,const double*v,const int*ci,const int*rs,const double*x,double beta,double*y){"
			+ "int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<rows){double s=0;for(int z=rs[row]-1;z<rs[row+1]-1;z++)s+=v[z]*x[ci[z]-1];y[row]=alpha*s+beta*y[row];}}"
			+ "extern \"C\" __global__ void csr_mm(int rows,int outcols,double alpha,const double*v,const int*ci,const int*rs,const double*b,double beta,double*c){"
			+ "int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<rows*outcols){int row=z/outcols,col=z-row*outcols;double s=0;"
			+ "for(int q=rs[row]-1;q<rs[row+1]-1;q++)s+=v[q]*b[(ci[q]-1)*outcols+col];c[z]=alpha*s+beta*c[z];}}"
			+ "extern \"C\" __global__ void float_blas_axpy(int n,float alpha,const float*x,int xo,int xs,float*y,int yo,int ys){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[yo+i*ys]=alpha*x[xo+i*xs]+y[yo+i*ys];}"
			+ "extern \"C\" __global__ void float_blas_dot(int n,const float*x,int xo,int xs,const float*y,int yo,int ys,float*out){"
			+ "if(blockIdx.x==0&&threadIdx.x==0){float s=0;for(int i=0;i<n;i++)s+=x[xo+i*xs]*y[yo+i*ys];out[0]=s;}}"
			+ "extern \"C\" __global__ void float_blas_nrm2(int n,const float*x,int xo,int xs,float*out){"
			+ "if(blockIdx.x==0&&threadIdx.x==0){float scale=0,sum=1;for(int i=0;i<n;i++){float v=fabsf(x[xo+i*xs]);"
			+ "if(v!=0){if(scale<v){float r=scale/v;sum=1+sum*r*r;scale=v;}else{float r=v/scale;sum+=r*r;}}}out[0]=scale==0?0:scale*sqrtf(sum);}}"
			+ "extern \"C\" __global__ void float_blas_gemv(int tr,int rows,int cols,float alpha,const float*a,const float*x,float beta,float*y){"
			+ "int i=blockIdx.x*blockDim.x+threadIdx.x,output=tr?cols:rows,input=tr?rows:cols;if(i<output){float s=0;"
			+ "for(int j=0;j<input;j++)s+=(tr?a[j*cols+i]:a[i*cols+j])*x[j];y[i]=alpha*s+beta*y[i];}}"
			+ "extern \"C\" __global__ void float_blas_gemm(int ta,int tb,int m,int n,int k,float alpha,const float*a,const float*b,float beta,float*c){"
			+ "int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;if(row<m&&col<n){float s=0;"
			+ "for(int q=0;q<k;q++)s+=(ta?a[q*m+row]:a[row*k+q])*(tb?b[col*k+q]:b[q*n+col]);int z=row*n+col;c[z]=alpha*s+beta*c[z];}}"
			+ "extern \"C\" __global__ void float_csr_mv(int rows,float alpha,const float*v,const int*ci,const int*rs,const float*x,float beta,float*y){"
			+ "int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<rows){float s=0;for(int z=rs[row]-1;z<rs[row+1]-1;z++)s+=v[z]*x[ci[z]-1];y[row]=alpha*s+beta*y[row];}}"
			+ "extern \"C\" __global__ void float_csr_mm(int rows,int outcols,float alpha,const float*v,const int*ci,const int*rs,const float*b,float beta,float*c){"
			+ "int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<rows*outcols){int row=z/outcols,col=z-row*outcols;float s=0;"
			+ "for(int q=rs[row]-1;q<rs[row+1]-1;q++)s+=v[q]*b[(ci[q]-1)*outcols+col];c[z]=alpha*s+beta*c[z];}}"
			+ "template<class T> __device__ T decomp_abs(T x){return x<0?-x:x;}"
			+ "template<class T> __device__ T decomp_hypot(T a,T b){a=decomp_abs(a);b=decomp_abs(b);T m=a>b?a:b;if(m==0)return 0;T x=a/m,y=b/m;return m*sqrt(x*x+y*y);}"
			+ "template<class T> __device__ void decomp_potrf(const T*a,T*l,int n,int*info){*info=0;for(int i=0;i<n*n;i++)l[i]=0;for(int r=0;r<n;r++)for(int c=0;c<=r;c++){T s=a[r*n+c];for(int k=0;k<c;k++)s-=l[r*n+k]*l[c*n+k];if(r==c){if(!(s>0)){*info=r+1;return;}l[r*n+c]=sqrt(s);}else l[r*n+c]=s/l[c*n+c];}}"
			+ "template<class T> __device__ T decomp_norm(const T*a,int rows,int cols,int col){T scale=0,sum=1;for(int r=0;r<rows;r++){T v=decomp_abs(a[r*cols+col]);if(v!=0){if(scale<v){T q=scale/v;sum=1+sum*q*q;scale=v;}else{T q=v/scale;sum+=q*q;}}}return scale==0?0:scale*sqrt(sum);}"
			+ "template<class T> __device__ void decomp_swapcols(T*a,int rows,int cols,int p,int q){for(int r=0;r<rows;r++){T v=a[r*cols+p];a[r*cols+p]=a[r*cols+q];a[r*cols+q]=v;}}"
			+ "template<class T> __device__ void decomp_qr(T*qr,T*tau,int*pivot,int rows,int cols){int count=rows<cols?rows:cols;for(int i=0;i<cols;i++)pivot[i]=i;for(int k=0;k<count;k++){int selected=k;T best=-1;for(int c=k;c<cols;c++){T norm=0;for(int r=k;r<rows;r++){T v=qr[r*cols+c];norm+=v*v;}if(norm>best){best=norm;selected=c;}}if(selected!=k){decomp_swapcols(qr,rows,cols,k,selected);int p=pivot[k];pivot[k]=pivot[selected];pivot[selected]=p;}T scale=0,sum=1;for(int r=k;r<rows;r++){T v=decomp_abs(qr[r*cols+k]);if(v!=0){if(scale<v){T q=scale/v;sum=1+sum*q*q;scale=v;}else{T q=v/scale;sum+=q*q;}}}T norm=scale==0?0:scale*sqrt(sum);if(norm==0){tau[k]=0;continue;}T alpha=qr[k*cols+k],diagonal=alpha<0?norm:-norm;tau[k]=(diagonal-alpha)/diagonal;T denominator=alpha-diagonal;qr[k*cols+k]=diagonal;for(int r=k+1;r<rows;r++)qr[r*cols+k]/=denominator;for(int c=k+1;c<cols;c++){T product=qr[k*cols+c];for(int r=k+1;r<rows;r++)product+=qr[r*cols+k]*qr[r*cols+c];product*=tau[k];qr[k*cols+c]-=product;for(int r=k+1;r<rows;r++)qr[r*cols+c]-=qr[r*cols+k]*product;}}}"
			+ "template<class T> __device__ void decomp_eigen(T*a,T*values,T*vectors,int n,T eps,int*info){*info=0;for(int i=0;i<n*n;i++)vectors[i]=0;for(int i=0;i<n;i++)vectors[i*n+i]=1;int maximum=8*n>32?8*n:32;for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int p=0;p<n-1;p++)for(int q=p+1;q<n;q++){T apq=a[p*n+q],threshold=eps*(decomp_abs(a[p*n+p])+decomp_abs(a[q*n+q])+1);if(decomp_abs(apq)<=threshold)continue;changed=true;T app=a[p*n+p],aqq=a[q*n+q],tau=(aqq-app)/(2*apq),t=(tau<0?-1:1)/(decomp_abs(tau)+decomp_hypot((T)1,tau)),c=1/decomp_hypot((T)1,t),s=t*c;for(int k=0;k<n;k++)if(k!=p&&k!=q){T akp=a[k*n+p],akq=a[k*n+q],np=c*akp-s*akq,nq=s*akp+c*akq;a[k*n+p]=a[p*n+k]=np;a[k*n+q]=a[q*n+k]=nq;}a[p*n+p]=c*c*app-2*s*c*apq+s*s*aqq;a[q*n+q]=s*s*app+2*s*c*apq+c*c*aqq;a[p*n+q]=a[q*n+p]=0;for(int r=0;r<n;r++){T vp=vectors[r*n+p],vq=vectors[r*n+q];vectors[r*n+p]=c*vp-s*vq;vectors[r*n+q]=s*vp+c*vq;}}if(!changed)break;if(sweep+1==maximum){*info=1;return;}}for(int i=0;i<n;i++)values[i]=a[i*n+i];for(int i=0;i<n-1;i++){int selected=i;for(int j=i+1;j<n;j++)if(values[j]<values[selected])selected=j;if(selected!=i){T z=values[i];values[i]=values[selected];values[selected]=z;decomp_swapcols(vectors,n,n,i,selected);}}for(int c=0;c<n;c++){int largest=0;for(int r=1;r<n;r++)if(decomp_abs(vectors[r*n+c])>decomp_abs(vectors[largest*n+c]))largest=r;if(vectors[largest*n+c]<0)for(int r=0;r<n;r++)vectors[r*n+c]=-vectors[r*n+c];}}"
			+ "template<class T> __device__ bool decomp_complete(T*u,int rows,int cols,int col,T eps){for(int candidate=0;candidate<rows;candidate++){for(int r=0;r<rows;r++)u[r*cols+col]=r==candidate?1:0;for(int p=0;p<col;p++){T product=0;for(int r=0;r<rows;r++)product+=u[r*cols+p]*u[r*cols+col];for(int r=0;r<rows;r++)u[r*cols+col]-=product*u[r*cols+p];}T norm=decomp_norm(u,rows,cols,col);if(norm>eps){for(int r=0;r<rows;r++)u[r*cols+col]/=norm;return true;}}return false;}"
			+ "template<class T> __device__ void decomp_svd(const T*a,T*work,T*u,T*singular,T*vt,int rows,int cols,T eps,int*info){*info=0;bool wide=rows<cols;int tr=wide?cols:rows,tc=wide?rows:cols;for(int r=0;r<tr;r++)for(int c=0;c<tc;c++)work[r*tc+c]=wide?a[c*cols+r]:a[r*cols+c];T*v=wide?u:vt;for(int i=0;i<tc*tc;i++)v[i]=0;for(int i=0;i<tc;i++)v[i*tc+i]=1;int maximum=12*tc>48?12*tc:48;for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int p=0;p<tc-1;p++)for(int q=p+1;q<tc;q++){T alpha=0,beta=0,gamma=0;for(int r=0;r<tr;r++){T x=work[r*tc+p],y=work[r*tc+q];alpha+=x*x;beta+=y*y;gamma+=x*y;}if(gamma==0||decomp_abs(gamma)<=eps*sqrt(alpha*beta))continue;changed=true;T zeta=(beta-alpha)/(2*gamma),t=(zeta<0?-1:1)/(decomp_abs(zeta)+decomp_hypot((T)1,zeta)),c=1/decomp_hypot((T)1,t),s=c*t;for(int r=0;r<tr;r++){int pi=r*tc+p,qi=r*tc+q;T x=work[pi],y=work[qi];work[pi]=c*x-s*y;work[qi]=s*x+c*y;}for(int r=0;r<tc;r++){int pi=r*tc+p,qi=r*tc+q;T x=v[pi],y=v[qi];v[pi]=c*x-s*y;v[qi]=s*x+c*y;}}if(!changed)break;if(sweep+1==maximum){*info=1;return;}}for(int c=0;c<tc;c++)singular[c]=decomp_norm(work,tr,tc,c);for(int i=0;i<tc-1;i++){int selected=i;for(int j=i+1;j<tc;j++)if(singular[j]>singular[selected])selected=j;if(selected!=i){T z=singular[i];singular[i]=singular[selected];singular[selected]=z;decomp_swapcols(work,tr,tc,i,selected);decomp_swapcols(v,tc,tc,i,selected);}}T threshold=(rows>cols?rows:cols)*eps/16*singular[0];if(!wide){for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)u[r*tc+c]=work[r*tc+c]/singular[c];else if(!decomp_complete(u,tr,tc,c,eps)){*info=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tr;r++)if(decomp_abs(u[r*tc+c])>decomp_abs(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tr;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tc;r++)v[r*tc+c]=-v[r*tc+c];}}for(int r=0;r<tc;r++)for(int c=r+1;c<tc;c++){T z=vt[r*tc+c];vt[r*tc+c]=vt[c*tc+r];vt[c*tc+r]=z;}}else{for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)work[r*tc+c]/=singular[c];else if(!decomp_complete(work,tr,tc,c,eps)){*info=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tc;r++)if(decomp_abs(u[r*tc+c])>decomp_abs(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tc;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tr;r++)work[r*tc+c]=-work[r*tc+c];}}for(int c=0;c<tc;c++)for(int r=0;r<tr;r++)vt[c*tr+r]=work[r*tc+c];}}"
			+ "extern \"C\" __global__ void decomp_dpotrf(const double*a,double*l,int n,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_potrf(a,l,n,info);}"
			+ "extern \"C\" __global__ void decomp_spotrf(const float*a,float*l,int n,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_potrf(a,l,n,info);}"
			+ "extern \"C\" __global__ void decomp_dgeqp3(double*qr,double*tau,int*pivot,int rows,int cols){if(blockIdx.x==0&&threadIdx.x==0)decomp_qr(qr,tau,pivot,rows,cols);}"
			+ "extern \"C\" __global__ void decomp_sgeqp3(float*qr,float*tau,int*pivot,int rows,int cols){if(blockIdx.x==0&&threadIdx.x==0)decomp_qr(qr,tau,pivot,rows,cols);}"
			+ "extern \"C\" __global__ void decomp_dsyev(double*a,double*values,double*vectors,int n,double eps,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_eigen(a,values,vectors,n,eps,info);}"
			+ "extern \"C\" __global__ void decomp_ssyev(float*a,float*values,float*vectors,int n,float eps,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_eigen(a,values,vectors,n,eps,info);}"
			+ "extern \"C\" __global__ void decomp_dgesvd(const double*a,double*w,double*u,double*s,double*vt,int rows,int cols,double eps,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_svd(a,w,u,s,vt,rows,cols,eps,info);}"
			+ "extern \"C\" __global__ void decomp_sgesvd(const float*a,float*w,float*u,float*s,float*vt,int rows,int cols,float eps,int*info){if(blockIdx.x==0&&threadIdx.x==0)decomp_svd(a,w,u,s,vt,rows,cols,eps,info);}"
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
	@Override public synchronized void daxpy(int count, double alpha, double[] x,
			int xOffset, int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return;
		ensureAvailable(); setCurrent(); CUdeviceptr dx = allocate(x), dy = allocate(y);
		try {
			launch("blas_axpy", grid(count), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {count}), Pointer.to(new double[] {alpha}), Pointer.to(dx),
					Pointer.to(new int[] {xOffset}), Pointer.to(new int[] {xStride}), Pointer.to(dy),
					Pointer.to(new int[] {yOffset}), Pointer.to(new int[] {yStride})));
			double[] updated = copy(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized double ddot(int count, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0;
		ensureAvailable(); setCurrent(); CUdeviceptr dx = allocate(x), dy = allocate(y), out = allocateDoubles(1);
		try {
			launch("blas_dot", 1, 1, 1, 1, 1, 1, Pointer.to(Pointer.to(new int[] {count}),
					Pointer.to(dx), Pointer.to(new int[] {xOffset}), Pointer.to(new int[] {xStride}),
					Pointer.to(dy), Pointer.to(new int[] {yOffset}), Pointer.to(new int[] {yStride}), Pointer.to(out)));
			return copy(out, 1)[0];
		} finally { cuMemFree(dx); cuMemFree(dy); cuMemFree(out); }
	}
	@Override public synchronized double dnrm2(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0; ensureAvailable(); setCurrent();
		CUdeviceptr dx = allocate(x), out = allocateDoubles(1);
		try {
			launch("blas_nrm2", 1, 1, 1, 1, 1, 1, Pointer.to(Pointer.to(new int[] {count}),
					Pointer.to(dx), Pointer.to(new int[] {offset}), Pointer.to(new int[] {stride}), Pointer.to(out)));
			return copy(out, 1)[0];
		} finally { cuMemFree(dx); cuMemFree(out); }
	}
	@Override public synchronized void dgemv(MatrixTranspose transpose, int rows, int columns,
			double alpha, double[] matrix, double[] x, double beta, double[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable(); setCurrent();
		CUdeviceptr a = allocate(matrix), dx = allocate(x), dy = allocate(y);
		try {
			int output = transpose == MatrixTranspose.NONE ? rows : columns;
			launch("blas_gemv", grid(output), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {transpose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rows}), Pointer.to(new int[] {columns}), Pointer.to(new double[] {alpha}),
					Pointer.to(a), Pointer.to(dx), Pointer.to(new double[] {beta}), Pointer.to(dy)));
			double[] updated = copy(dy, output); System.arraycopy(updated, 0, y, 0, output);
		} finally { cuMemFree(a); cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left, double[] right,
			double beta, double[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result);
		ensureAvailable(); setCurrent(); CUdeviceptr a = allocate(left), b = allocate(right), c = allocate(result);
		try {
			launch("blas_gemm", (columns + 15) / 16, (rows + 15) / 16, 1, 16, 16, 1, Pointer.to(
					Pointer.to(new int[] {leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rows}), Pointer.to(new int[] {columns}), Pointer.to(new int[] {shared}),
					Pointer.to(new double[] {alpha}), Pointer.to(a), Pointer.to(b), Pointer.to(new double[] {beta}), Pointer.to(c)));
			double[] updated = copy(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { cuMemFree(a); cuMemFree(b); cuMemFree(c); }
	}
	@Override public synchronized void dcsrmv(double alpha, CsrMatrix matrix, double[] x,
			double beta, double[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable(); setCurrent();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		CUdeviceptr values = allocate(matrix.values()), indices = allocate(matrix.columnIndices());
		CUdeviceptr starts = allocate(matrix.rowStarts()), dx = allocate(x), dy = allocate(y);
		try {
			launch("csr_mv", grid(matrix.rows()), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {matrix.rows()}), Pointer.to(new double[] {alpha}), Pointer.to(values),
					Pointer.to(indices), Pointer.to(starts), Pointer.to(dx), Pointer.to(new double[] {beta}), Pointer.to(dy)));
			double[] updated = copy(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { cuMemFree(values); cuMemFree(indices); cuMemFree(starts); cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable(); setCurrent();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		CUdeviceptr values = allocate(matrix.values()), indices = allocate(matrix.columnIndices());
		CUdeviceptr starts = allocate(matrix.rowStarts()), b = allocate(right), c = allocate(result);
		try {
			launch("csr_mm", grid(result.length), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {matrix.rows()}), Pointer.to(new int[] {rightColumns}),
					Pointer.to(new double[] {alpha}), Pointer.to(values), Pointer.to(indices), Pointer.to(starts),
					Pointer.to(b), Pointer.to(new double[] {beta}), Pointer.to(c)));
			double[] updated = copy(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { cuMemFree(values); cuMemFree(indices); cuMemFree(starts); cuMemFree(b); cuMemFree(c); }
	}
	@Override public synchronized void saxpy(int count, float alpha, float[] x,
			int xOffset, int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return;
		ensureAvailable(); setCurrent(); CUdeviceptr dx = allocate(x), dy = allocate(y);
		try {
			launch("float_blas_axpy", grid(count), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {count}), Pointer.to(new float[] {alpha}), Pointer.to(dx),
					Pointer.to(new int[] {xOffset}), Pointer.to(new int[] {xStride}), Pointer.to(dy),
					Pointer.to(new int[] {yOffset}), Pointer.to(new int[] {yStride})));
			float[] updated = copyFloats(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized float sdot(int count, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0f;
		ensureAvailable(); setCurrent(); CUdeviceptr dx = allocate(x), dy = allocate(y), out = allocateFloats(1);
		try {
			launch("float_blas_dot", 1, 1, 1, 1, 1, 1, Pointer.to(Pointer.to(new int[] {count}),
					Pointer.to(dx), Pointer.to(new int[] {xOffset}), Pointer.to(new int[] {xStride}),
					Pointer.to(dy), Pointer.to(new int[] {yOffset}), Pointer.to(new int[] {yStride}), Pointer.to(out)));
			return copyFloats(out, 1)[0];
		} finally { cuMemFree(dx); cuMemFree(dy); cuMemFree(out); }
	}
	@Override public synchronized float snrm2(int count, float[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0f; ensureAvailable(); setCurrent();
		CUdeviceptr dx = allocate(x), out = allocateFloats(1);
		try {
			launch("float_blas_nrm2", 1, 1, 1, 1, 1, 1, Pointer.to(Pointer.to(new int[] {count}),
					Pointer.to(dx), Pointer.to(new int[] {offset}), Pointer.to(new int[] {stride}), Pointer.to(out)));
			return copyFloats(out, 1)[0];
		} finally { cuMemFree(dx); cuMemFree(out); }
	}
	@Override public synchronized void sgemv(MatrixTranspose transpose, int rows, int columns,
			float alpha, float[] matrix, float[] x, float beta, float[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable(); setCurrent();
		CUdeviceptr a = allocate(matrix), dx = allocate(x), dy = allocate(y);
		try {
			int output = transpose == MatrixTranspose.NONE ? rows : columns;
			launch("float_blas_gemv", grid(output), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {transpose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rows}), Pointer.to(new int[] {columns}), Pointer.to(new float[] {alpha}),
					Pointer.to(a), Pointer.to(dx), Pointer.to(new float[] {beta}), Pointer.to(dy)));
			float[] updated = copyFloats(dy, output); System.arraycopy(updated, 0, y, 0, output);
		} finally { cuMemFree(a); cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left, float[] right,
			float beta, float[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result);
		ensureAvailable(); setCurrent(); CUdeviceptr a = allocate(left), b = allocate(right), c = allocate(result);
		try {
			launch("float_blas_gemm", (columns + 15) / 16, (rows + 15) / 16, 1, 16, 16, 1, Pointer.to(
					Pointer.to(new int[] {leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0}),
					Pointer.to(new int[] {rows}), Pointer.to(new int[] {columns}), Pointer.to(new int[] {shared}),
					Pointer.to(new float[] {alpha}), Pointer.to(a), Pointer.to(b), Pointer.to(new float[] {beta}), Pointer.to(c)));
			float[] updated = copyFloats(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { cuMemFree(a); cuMemFree(b); cuMemFree(c); }
	}
	@Override public synchronized void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x,
			float beta, float[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable(); setCurrent();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		CUdeviceptr values = allocate(matrix.values()), indices = allocate(matrix.columnIndices());
		CUdeviceptr starts = allocate(matrix.rowStarts()), dx = allocate(x), dy = allocate(y);
		try {
			launch("float_csr_mv", grid(matrix.rows()), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {matrix.rows()}), Pointer.to(new float[] {alpha}), Pointer.to(values),
					Pointer.to(indices), Pointer.to(starts), Pointer.to(dx), Pointer.to(new float[] {beta}), Pointer.to(dy)));
			float[] updated = copyFloats(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { cuMemFree(values); cuMemFree(indices); cuMemFree(starts); cuMemFree(dx); cuMemFree(dy); }
	}
	@Override public synchronized void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable(); setCurrent();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		CUdeviceptr values = allocate(matrix.values()), indices = allocate(matrix.columnIndices());
		CUdeviceptr starts = allocate(matrix.rowStarts()), b = allocate(right), c = allocate(result);
		try {
			launch("float_csr_mm", grid(result.length), 1, 1, BLOCK, 1, 1, Pointer.to(
					Pointer.to(new int[] {matrix.rows()}), Pointer.to(new int[] {rightColumns}),
					Pointer.to(new float[] {alpha}), Pointer.to(values), Pointer.to(indices), Pointer.to(starts),
					Pointer.to(b), Pointer.to(new float[] {beta}), Pointer.to(c)));
			float[] updated = copyFloats(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { cuMemFree(values); cuMemFree(indices); cuMemFree(starts); cuMemFree(b); cuMemFree(c); }
	}
	@Override public synchronized CholeskyFactor dpotrf(double[] matrix, int dimension) {
		checkDecompositionMatrix(matrix, dimension, dimension); ensureAvailable(); setCurrent();
		CUdeviceptr a=allocate(matrix), lower=allocateDoubles(matrix.length), info=allocateInts(1);
		try { launch("decomp_dpotrf",1,1,1,1,1,1,Pointer.to(Pointer.to(a),Pointer.to(lower),Pointer.to(new int[]{dimension}),Pointer.to(info)));
			int status=copyInts(info,1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);
			return new CholeskyFactor(dimension,copy(lower,matrix.length));
		} finally {cuMemFree(info);cuMemFree(lower);cuMemFree(a);}
	}
	@Override public synchronized PivotedQrFactor dgeqp3(double[] matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();setCurrent();int count=Math.min(rows,columns);CUdeviceptr qr=allocate(matrix),tau=allocateDoubles(count),pivot=allocateInts(columns);try{launch("decomp_dgeqp3",1,1,1,1,1,1,Pointer.to(Pointer.to(qr),Pointer.to(tau),Pointer.to(pivot),Pointer.to(new int[]{rows}),Pointer.to(new int[]{columns})));return new PivotedQrFactor(rows,columns,copy(qr,matrix.length),copy(tau,count),copyInts(pivot,columns));}finally{cuMemFree(pivot);cuMemFree(tau);cuMemFree(qr);}}
	@Override public synchronized SymmetricEigenDecomposition dsyev(double[] matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();setCurrent();CUdeviceptr work=allocate(matrix),values=allocateDoubles(dimension),vectors=allocateDoubles(matrix.length),info=allocateInts(1);try{launch("decomp_dsyev",1,1,1,1,1,1,Pointer.to(Pointer.to(work),Pointer.to(values),Pointer.to(vectors),Pointer.to(new int[]{dimension}),Pointer.to(new double[]{16*Math.ulp(1.0)}),Pointer.to(info)));if(copyInts(info,1)[0]!=0)throw new IllegalStateException("CUDA symmetric eigendecomposition did not converge");return new SymmetricEigenDecomposition(dimension,copy(values,dimension),copy(vectors,matrix.length));}finally{cuMemFree(info);cuMemFree(vectors);cuMemFree(values);cuMemFree(work);}}
	@Override public synchronized SingularValueDecomposition dgesvd(double[] matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();setCurrent();int count=Math.min(rows,columns);CUdeviceptr a=allocate(matrix),work=allocateDoubles(matrix.length),u=allocateDoubles(rows*count),singular=allocateDoubles(count),vt=allocateDoubles(count*columns),info=allocateInts(1);try{launch("decomp_dgesvd",1,1,1,1,1,1,Pointer.to(Pointer.to(a),Pointer.to(work),Pointer.to(u),Pointer.to(singular),Pointer.to(vt),Pointer.to(new int[]{rows}),Pointer.to(new int[]{columns}),Pointer.to(new double[]{16*Math.ulp(1.0)}),Pointer.to(info)));if(copyInts(info,1)[0]!=0)throw new IllegalStateException("CUDA SVD did not converge");return new SingularValueDecomposition(rows,columns,copy(singular,count),copy(u,rows*count),copy(vt,count*columns));}finally{cuMemFree(info);cuMemFree(vt);cuMemFree(singular);cuMemFree(u);cuMemFree(work);cuMemFree(a);}}
	@Override public synchronized FloatCholeskyFactor spotrf(float[] matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);ensureAvailable();setCurrent();CUdeviceptr a=allocate(matrix),lower=allocateFloats(matrix.length),info=allocateInts(1);try{launch("decomp_spotrf",1,1,1,1,1,1,Pointer.to(Pointer.to(a),Pointer.to(lower),Pointer.to(new int[]{dimension}),Pointer.to(info)));int status=copyInts(info,1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);return new FloatCholeskyFactor(dimension,copyFloats(lower,matrix.length));}finally{cuMemFree(info);cuMemFree(lower);cuMemFree(a);}}
	@Override public synchronized FloatPivotedQrFactor sgeqp3(float[] matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();setCurrent();int count=Math.min(rows,columns);CUdeviceptr qr=allocate(matrix),tau=allocateFloats(count),pivot=allocateInts(columns);try{launch("decomp_sgeqp3",1,1,1,1,1,1,Pointer.to(Pointer.to(qr),Pointer.to(tau),Pointer.to(pivot),Pointer.to(new int[]{rows}),Pointer.to(new int[]{columns})));return new FloatPivotedQrFactor(rows,columns,copyFloats(qr,matrix.length),copyFloats(tau,count),copyInts(pivot,columns));}finally{cuMemFree(pivot);cuMemFree(tau);cuMemFree(qr);}}
	@Override public synchronized FloatSymmetricEigenDecomposition ssyev(float[] matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();setCurrent();CUdeviceptr work=allocate(matrix),values=allocateFloats(dimension),vectors=allocateFloats(matrix.length),info=allocateInts(1);try{launch("decomp_ssyev",1,1,1,1,1,1,Pointer.to(Pointer.to(work),Pointer.to(values),Pointer.to(vectors),Pointer.to(new int[]{dimension}),Pointer.to(new float[]{16*Math.ulp(1.0f)}),Pointer.to(info)));if(copyInts(info,1)[0]!=0)throw new IllegalStateException("CUDA FP32 symmetric eigendecomposition did not converge");return new FloatSymmetricEigenDecomposition(dimension,copyFloats(values,dimension),copyFloats(vectors,matrix.length));}finally{cuMemFree(info);cuMemFree(vectors);cuMemFree(values);cuMemFree(work);}}
	@Override public synchronized FloatSingularValueDecomposition sgesvd(float[] matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();setCurrent();int count=Math.min(rows,columns);CUdeviceptr a=allocate(matrix),work=allocateFloats(matrix.length),u=allocateFloats(rows*count),singular=allocateFloats(count),vt=allocateFloats(count*columns),info=allocateInts(1);try{launch("decomp_sgesvd",1,1,1,1,1,1,Pointer.to(Pointer.to(a),Pointer.to(work),Pointer.to(u),Pointer.to(singular),Pointer.to(vt),Pointer.to(new int[]{rows}),Pointer.to(new int[]{columns}),Pointer.to(new float[]{16*Math.ulp(1.0f)}),Pointer.to(info)));if(copyInts(info,1)[0]!=0)throw new IllegalStateException("CUDA FP32 SVD did not converge");return new FloatSingularValueDecomposition(rows,columns,copyFloats(singular,count),copyFloats(u,rows*count),copyFloats(vt,count*columns));}finally{cuMemFree(info);cuMemFree(vt);cuMemFree(singular);cuMemFree(u);cuMemFree(work);cuMemFree(a);}}
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
				major[0] >= 2, true, memory[0], true, true, true);
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
	private static CUdeviceptr allocate(float[] values) {
		CUdeviceptr result = allocateFloats(values.length);
		cuMemcpyHtoD(result, Pointer.to(values), (long) values.length * Sizeof.FLOAT); return result;
	}
	private static CUdeviceptr allocate(int[] values) {
		CUdeviceptr result = new CUdeviceptr(); cuMemAlloc(result, (long) values.length * Sizeof.INT);
		cuMemcpyHtoD(result, Pointer.to(values), (long) values.length * Sizeof.INT); return result;
	}
	private static CUdeviceptr allocateDoubles(int count) {
		CUdeviceptr result = new CUdeviceptr(); cuMemAlloc(result, (long) count * Sizeof.DOUBLE); return result;
	}
	private static CUdeviceptr allocateFloats(int count) {
		CUdeviceptr result = new CUdeviceptr(); cuMemAlloc(result, (long) count * Sizeof.FLOAT); return result;
	}
	private static CUdeviceptr allocateInts(int count) {
		CUdeviceptr result = new CUdeviceptr(); cuMemAlloc(result, (long) count * Sizeof.INT); return result;
	}
	private static double[] copy(CUdeviceptr source, int count) {
		double[] result = new double[count]; cuMemcpyDtoH(Pointer.to(result), source, (long) count * Sizeof.DOUBLE); return result;
	}
	private static float[] copyFloats(CUdeviceptr source, int count) {
		float[] result = new float[count]; cuMemcpyDtoH(Pointer.to(result), source, (long) count * Sizeof.FLOAT); return result;
	}
	private static int[] copyInts(CUdeviceptr source, int count) {
		int[] result = new int[count]; cuMemcpyDtoH(Pointer.to(result), source, (long) count * Sizeof.INT); return result;
	}
	private static int grid(int count) { return (count + BLOCK - 1) / BLOCK; }
	private void setCurrent() { cuCtxSetCurrent(context); }
	private void ensureAvailable() {
		if (!available()) throw new IllegalStateException("CUDA backend unavailable", unavailableCause);
	}
	private static void checkVectors(double[] x, double[] y) {
		if (x == null || y == null || x.length != y.length) throw new IllegalArgumentException("vector lengths must match");
	}
	private static void checkRegion(int count, double[] values, int offset, int stride) {
		if (count < 0 || values == null || offset < 0 || stride < 1
				|| (count > 0 && (long) offset + (long) (count - 1) * stride >= values.length))
			throw new IllegalArgumentException("invalid strided vector region");
	}
	private static void checkRegion(int count, float[] values, int offset, int stride) {
		if (count < 0 || values == null || offset < 0 || stride < 1
				|| (count > 0 && (long) offset + (long) (count - 1) * stride >= values.length))
			throw new IllegalArgumentException("invalid strided vector region");
	}
	private static void checkGemv(MatrixTranspose transpose, int rows, int columns,
			double[] matrix, double[] x, double[] y) {
		if (transpose == null || rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns)
			throw new IllegalArgumentException("invalid GEMV matrix");
		if (x == null || x.length != (transpose == MatrixTranspose.NONE ? columns : rows)
				|| y == null || y.length != (transpose == MatrixTranspose.NONE ? rows : columns))
			throw new IllegalArgumentException("GEMV vector dimensions do not conform");
	}
	private static void checkGemv(MatrixTranspose transpose, int rows, int columns,
			float[] matrix, float[] x, float[] y) {
		if (transpose == null || rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns)
			throw new IllegalArgumentException("invalid GEMV matrix");
		if (x == null || x.length != (transpose == MatrixTranspose.NONE ? columns : rows)
				|| y == null || y.length != (transpose == MatrixTranspose.NONE ? rows : columns))
			throw new IllegalArgumentException("GEMV vector dimensions do not conform");
	}
	private static void checkGemm(MatrixTranspose ta, MatrixTranspose tb, int m, int n, int k,
			double[] a, double[] b, double[] c) {
		if (ta == null || tb == null || m < 1 || n < 1 || k < 1 || a == null || b == null || c == null
				|| a.length != m * k || b.length != k * n || c.length != m * n)
			throw new IllegalArgumentException("GEMM dimensions do not conform");
	}
	private static void checkGemm(MatrixTranspose ta, MatrixTranspose tb, int m, int n, int k,
			float[] a, float[] b, float[] c) {
		if (ta == null || tb == null || m < 1 || n < 1 || k < 1 || a == null || b == null || c == null
				|| a.length != m * k || b.length != k * n || c.length != m * n)
			throw new IllegalArgumentException("GEMM dimensions do not conform");
	}
	private static void checkCsrMv(CsrMatrix matrix, double[] x, double[] y) {
		if (matrix == null || x == null || x.length != matrix.columns() || y == null || y.length != matrix.rows())
			throw new IllegalArgumentException("CSR matrix-vector dimensions do not conform");
	}
	private static void checkCsrMm(CsrMatrix matrix, double[] right, int columns, double[] result) {
		if (matrix == null || columns < 1 || right == null || right.length != matrix.columns() * columns
				|| result == null || result.length != matrix.rows() * columns)
			throw new IllegalArgumentException("CSR matrix-matrix dimensions do not conform");
	}
	private static void checkCsrMv(FloatCsrMatrix matrix, float[] x, float[] y) {
		if (matrix == null || x == null || x.length != matrix.columns() || y == null || y.length != matrix.rows())
			throw new IllegalArgumentException("CSR matrix-vector dimensions do not conform");
	}
	private static void checkCsrMm(FloatCsrMatrix matrix, float[] right, int columns, float[] result) {
		if (matrix == null || columns < 1 || right == null || right.length != matrix.columns() * columns
				|| result == null || result.length != matrix.rows() * columns)
			throw new IllegalArgumentException("CSR matrix-matrix dimensions do not conform");
	}
	private static void checkDecompositionMatrix(double[] matrix,int rows,int columns){
		if(rows<1||columns<1||matrix==null||matrix.length!=rows*columns)throw new IllegalArgumentException("invalid decomposition matrix dimensions");
		for(double value:matrix)if(!Double.isFinite(value))throw new IllegalArgumentException("decomposition matrix must be finite");
	}
	private static void checkDecompositionMatrix(float[] matrix,int rows,int columns){
		if(rows<1||columns<1||matrix==null||matrix.length!=rows*columns)throw new IllegalArgumentException("invalid FP32 decomposition matrix dimensions");
		for(float value:matrix)if(!Float.isFinite(value))throw new IllegalArgumentException("FP32 decomposition matrix must be finite");
	}
	private static void checkSymmetric(double[]matrix,int n){double scale=1;for(double value:matrix)scale=Math.max(scale,Math.abs(value));double tolerance=64*Math.ulp(1.0)*scale;for(int r=0;r<n;r++)for(int c=r+1;c<n;c++)if(Math.abs(matrix[r*n+c]-matrix[c*n+r])>tolerance)throw new IllegalArgumentException("eigenvalue matrix must be symmetric");}
	private static void checkSymmetric(float[]matrix,int n){float scale=1;for(float value:matrix)scale=Math.max(scale,Math.abs(value));float tolerance=64*Math.ulp(1.0f)*scale;for(int r=0;r<n;r++)for(int c=r+1;c<n;c++)if(Math.abs(matrix[r*n+c]-matrix[c*n+r])>tolerance)throw new IllegalArgumentException("FP32 eigenvalue matrix must be symmetric");}
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
