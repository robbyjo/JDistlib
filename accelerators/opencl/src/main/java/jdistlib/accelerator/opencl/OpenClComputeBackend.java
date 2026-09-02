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
import jdistlib.accelerator.ComputeApi;
import jdistlib.accelerator.ComputeDeviceInfo;
import jdistlib.accelerator.CholeskyFactor;
import jdistlib.accelerator.FloatCholeskyFactor;
import jdistlib.accelerator.FloatPivotedQrFactor;
import jdistlib.accelerator.FloatSingularValueDecomposition;
import jdistlib.accelerator.FloatSymmetricEigenDecomposition;
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.MatrixTranspose;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.MatrixDiagonal;
import jdistlib.accelerator.MatrixSide;
import jdistlib.accelerator.PreparedLogisticRegression;
import jdistlib.accelerator.PreparedDenseMatrix;
import jdistlib.accelerator.PreparedFloatDenseMatrix;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.accelerator.PreparedTransposeProduct;
import jdistlib.accelerator.PivotedQrFactor;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.SparseCholeskyPlan;
import jdistlib.accelerator.SparseOrdering;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** Optional portable OpenCL 1.2 backend using JOCL. */
public final class OpenClComputeBackend implements ComputeBackend {
	private static final long LOCAL = 256L;
	private static final String DECOMPOSITION_TEMPLATE =
			"$T $Pabs($T x){return x<0?-x:x;}\n"
			+ "$T $Phypot($T a,$T b){a=$Pabs(a);b=$Pabs(b);$T m=a>b?a:b;if(m==0)return 0;$T x=a/m,y=b/m;return m*sqrt(x*x+y*y);}\n"
			+ "$T $Pnorm(__global const $T*a,int rows,int cols,int col){$T scale=0,sum=1;for(int r=0;r<rows;r++){$T v=$Pabs(a[r*cols+col]);if(v!=0){if(scale<v){$T q=scale/v;sum=1+sum*q*q;scale=v;}else{$T q=v/scale;sum+=q*q;}}}return scale==0?0:scale*sqrt(sum);}\n"
			+ "void $Pswap(__global $T*a,int rows,int cols,int p,int q){for(int r=0;r<rows;r++){$T v=a[r*cols+p];a[r*cols+p]=a[r*cols+q];a[r*cols+q]=v;}}\n"
			+ "bool $Pcomplete(__global $T*u,int rows,int cols,int col,$T eps){for(int candidate=0;candidate<rows;candidate++){for(int r=0;r<rows;r++)u[r*cols+col]=r==candidate?1:0;for(int p=0;p<col;p++){$T product=0;for(int r=0;r<rows;r++)product+=u[r*cols+p]*u[r*cols+col];for(int r=0;r<rows;r++)u[r*cols+col]-=product*u[r*cols+p];}$T norm=$Pnorm(u,rows,cols,col);if(norm>eps){for(int r=0;r<rows;r++)u[r*cols+col]/=norm;return true;}}return false;}\n"
			+ "__kernel void decomp_$Ppotrf(__global const $T*a,__global $T*l,int n,__global int*info){if(get_global_id(0)!=0)return;*info=0;for(int i=0;i<n*n;i++)l[i]=0;for(int r=0;r<n;r++)for(int c=0;c<=r;c++){$T s=a[r*n+c];for(int k=0;k<c;k++)s-=l[r*n+k]*l[c*n+k];if(r==c){if(!(s>0)){*info=r+1;return;}l[r*n+c]=sqrt(s);}else l[r*n+c]=s/l[c*n+c];}}\n"
			+ "__kernel void decomp_$Pgeqp3(__global $T*qr,__global $T*tau,__global int*pivot,int rows,int cols){if(get_global_id(0)!=0)return;int count=rows<cols?rows:cols;for(int i=0;i<cols;i++)pivot[i]=i;for(int k=0;k<count;k++){int selected=k;$T best=-1;for(int c=k;c<cols;c++){$T norm=0;for(int r=k;r<rows;r++){$T v=qr[r*cols+c];norm+=v*v;}if(norm>best){best=norm;selected=c;}}if(selected!=k){$Pswap(qr,rows,cols,k,selected);int z=pivot[k];pivot[k]=pivot[selected];pivot[selected]=z;}$T scale=0,sum=1;for(int r=k;r<rows;r++){$T v=$Pabs(qr[r*cols+k]);if(v!=0){if(scale<v){$T z=scale/v;sum=1+sum*z*z;scale=v;}else{$T z=v/scale;sum+=z*z;}}}$T norm=scale==0?0:scale*sqrt(sum);if(norm==0){tau[k]=0;continue;}$T alpha=qr[k*cols+k],diagonal=alpha<0?norm:-norm;tau[k]=(diagonal-alpha)/diagonal;$T denominator=alpha-diagonal;qr[k*cols+k]=diagonal;for(int r=k+1;r<rows;r++)qr[r*cols+k]/=denominator;for(int c=k+1;c<cols;c++){$T product=qr[k*cols+c];for(int r=k+1;r<rows;r++)product+=qr[r*cols+k]*qr[r*cols+c];product*=tau[k];qr[k*cols+c]-=product;for(int r=k+1;r<rows;r++)qr[r*cols+c]-=qr[r*cols+k]*product;}}}\n"
			+ "__kernel void decomp_$Psyev(__global $T*a,__global $T*values,__global $T*vectors,int n,$T eps,__global int*info){if(get_global_id(0)!=0)return;*info=0;for(int i=0;i<n*n;i++)vectors[i]=0;for(int i=0;i<n;i++)vectors[i*n+i]=1;int maximum=8*n>32?8*n:32;for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int p=0;p<n-1;p++)for(int q=p+1;q<n;q++){$T apq=a[p*n+q],threshold=eps*($Pabs(a[p*n+p])+$Pabs(a[q*n+q])+1);if($Pabs(apq)<=threshold)continue;changed=true;$T app=a[p*n+p],aqq=a[q*n+q],tau=(aqq-app)/(2*apq),t=(tau<0?-1:1)/($Pabs(tau)+$Phypot(($T)1,tau)),c=1/$Phypot(($T)1,t),s=t*c;for(int k=0;k<n;k++)if(k!=p&&k!=q){$T akp=a[k*n+p],akq=a[k*n+q],np=c*akp-s*akq,nq=s*akp+c*akq;a[k*n+p]=a[p*n+k]=np;a[k*n+q]=a[q*n+k]=nq;}a[p*n+p]=c*c*app-2*s*c*apq+s*s*aqq;a[q*n+q]=s*s*app+2*s*c*apq+c*c*aqq;a[p*n+q]=a[q*n+p]=0;for(int r=0;r<n;r++){$T vp=vectors[r*n+p],vq=vectors[r*n+q];vectors[r*n+p]=c*vp-s*vq;vectors[r*n+q]=s*vp+c*vq;}}if(!changed)break;if(sweep+1==maximum){*info=1;return;}}for(int i=0;i<n;i++)values[i]=a[i*n+i];for(int i=0;i<n-1;i++){int selected=i;for(int j=i+1;j<n;j++)if(values[j]<values[selected])selected=j;if(selected!=i){$T z=values[i];values[i]=values[selected];values[selected]=z;$Pswap(vectors,n,n,i,selected);}}for(int c=0;c<n;c++){int largest=0;for(int r=1;r<n;r++)if($Pabs(vectors[r*n+c])>$Pabs(vectors[largest*n+c]))largest=r;if(vectors[largest*n+c]<0)for(int r=0;r<n;r++)vectors[r*n+c]=-vectors[r*n+c];}}\n"
			+ "__kernel void decomp_$Pgesvd(__global const $T*a,__global $T*work,__global $T*u,__global $T*singular,__global $T*vt,int rows,int cols,$T eps,__global int*info){if(get_global_id(0)!=0)return;*info=0;bool wide=rows<cols;int tr=wide?cols:rows,tc=wide?rows:cols;for(int r=0;r<tr;r++)for(int c=0;c<tc;c++)work[r*tc+c]=wide?a[c*cols+r]:a[r*cols+c];__global $T*v=wide?u:vt;for(int i=0;i<tc*tc;i++)v[i]=0;for(int i=0;i<tc;i++)v[i*tc+i]=1;int maximum=12*tc>48?12*tc:48;for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int p=0;p<tc-1;p++)for(int q=p+1;q<tc;q++){$T alpha=0,beta=0,gamma=0;for(int r=0;r<tr;r++){$T x=work[r*tc+p],y=work[r*tc+q];alpha+=x*x;beta+=y*y;gamma+=x*y;}if(gamma==0||$Pabs(gamma)<=eps*sqrt(alpha*beta))continue;changed=true;$T zeta=(beta-alpha)/(2*gamma),t=(zeta<0?-1:1)/($Pabs(zeta)+$Phypot(($T)1,zeta)),c=1/$Phypot(($T)1,t),s=c*t;for(int r=0;r<tr;r++){int pi=r*tc+p,qi=r*tc+q;$T x=work[pi],y=work[qi];work[pi]=c*x-s*y;work[qi]=s*x+c*y;}for(int r=0;r<tc;r++){int pi=r*tc+p,qi=r*tc+q;$T x=v[pi],y=v[qi];v[pi]=c*x-s*y;v[qi]=s*x+c*y;}}if(!changed)break;if(sweep+1==maximum){*info=1;return;}}for(int c=0;c<tc;c++)singular[c]=$Pnorm(work,tr,tc,c);for(int i=0;i<tc-1;i++){int selected=i;for(int j=i+1;j<tc;j++)if(singular[j]>singular[selected])selected=j;if(selected!=i){$T z=singular[i];singular[i]=singular[selected];singular[selected]=z;$Pswap(work,tr,tc,i,selected);$Pswap(v,tc,tc,i,selected);}}$T threshold=(rows>cols?rows:cols)*eps/16*singular[0];if(!wide){for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)u[r*tc+c]=work[r*tc+c]/singular[c];else if(!$Pcomplete(u,tr,tc,c,eps)){*info=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tr;r++)if($Pabs(u[r*tc+c])>$Pabs(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tr;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tc;r++)v[r*tc+c]=-v[r*tc+c];}}for(int r=0;r<tc;r++)for(int c=r+1;c<tc;c++){$T z=vt[r*tc+c];vt[r*tc+c]=vt[c*tc+r];vt[c*tc+r]=z;}}else{for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)work[r*tc+c]/=singular[c];else if(!$Pcomplete(work,tr,tc,c,eps)){*info=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tc;r++)if($Pabs(u[r*tc+c])>$Pabs(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tc;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tr;r++)work[r*tc+c]=-work[r*tc+c];}}for(int c=0;c<tc;c++)for(int r=0;r<tr;r++)vt[c*tr+r]=work[r*tc+c];}}\n";
	private static String decompositionSource(String type,String prefix){return DECOMPOSITION_TEMPLATE.replace("$T",type).replace("$P",prefix);}
	private static final String SPARSE_TEMPLATE =
			"__kernel void sparse_$Ppotrf(__global const $T*a,__global $T*l,__global const int*ci,__global const int*rs,int n,__global int*info,__global $T*logdet){if(get_global_id(0)!=0)return;*info=0;$T determinant=0;for(int row=0;row<n;row++){for(int at=rs[row];at<rs[row+1];at++){int col=ci[at];$T sum=a[at];int left=rs[row],other=rs[col];while(left<at&&other<rs[col+1]){int lc=ci[left],oc=ci[other];if(lc>=col||oc>=col)break;if(lc==oc){sum-=l[left]*l[other];left++;other++;}else if(lc<oc)left++;else other++;}if(row==col){if(!(sum>0)||!isfinite(sum)){*info=row+1;return;}l[at]=sqrt(sum);determinant+=2*log(l[at]);}else l[at]=sum/l[rs[col+1]-1];}}*logdet=determinant;}\n"
			+ "__kernel void sparse_$Psolve(__global const $T*l,__global const int*ci,__global const int*rs,__global const int*perm,__global const $T*right,__global $T*work,__global $T*result,int n,int columns){int rhs=get_global_id(0);if(rhs>=columns)return;for(int row=0;row<n;row++)work[row*columns+rhs]=right[perm[row]*columns+rhs];for(int row=0;row<n;row++){$T value=work[row*columns+rhs];int end=rs[row+1]-1;for(int at=rs[row];at<end;at++)value-=l[at]*work[ci[at]*columns+rhs];work[row*columns+rhs]=value/l[end];}for(int row=n-1;row>=0;row--){int end=rs[row+1]-1;work[row*columns+rhs]/=l[end];for(int at=rs[row];at<end;at++)work[ci[at]*columns+rhs]-=l[at]*work[row*columns+rhs];}for(int row=0;row<n;row++)result[perm[row]*columns+rhs]=work[row*columns+rhs];}\n";
	private static String sparseSource(String type,String prefix){return SPARSE_TEMPLATE.replace("$T",type).replace("$P",prefix);}
	private static final String SOURCE =
			"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
			+ "double logistic(double v){return v>=0?1.0/(1.0+exp(-v)):exp(v)/(1.0+exp(v));}\n"
			+ "double l1e(double v){return v>0?v+log1p(exp(-v)):log1p(exp(v));}\n"
			+ "__kernel void unary_kernel(int op,__global const double*x,__global double*y,int n){int i=get_global_id(0);if(i<n){double v=x[i];if(op==0)y[i]=exp(v);else if(op==1)y[i]=log(v);else if(op==2)y[i]=log1p(v);else if(op==3)y[i]=sqrt(v);else if(op==4)y[i]=tanh(v);else y[i]=logistic(v);}}\n"
			+ "__kernel void axpy_kernel(double a,__global const double*x,__global const double*y,__global double*z,int n){int i=get_global_id(0);if(i<n)z[i]=a*x[i]+y[i];}\n"
			+ "__kernel void dot_kernel(__global const double*x,__global const double*y,__global double*out,int n){if(get_global_id(0)==0){double s=0;for(int i=0;i<n;i++)s+=x[i]*y[i];out[0]=s;}}\n"
			+ "__kernel void gemm_kernel(__global const double*a,__global const double*b,__global double*c,int m,int k,int n){int row=get_global_id(1),col=get_global_id(0);if(row<m&&col<n){double s=0;for(int p=0;p<k;p++)s+=a[row*k+p]*b[p*n+col];c[row*n+col]=s;}}\n"
			+ "__kernel void blas_axpy(int n,double alpha,__global const double*x,int xo,int xs,__global double*y,int yo,int ys){int i=get_global_id(0);if(i<n)y[yo+i*ys]=alpha*x[xo+i*xs]+y[yo+i*ys];}\n"
			+ "__kernel void blas_dot(int n,__global const double*x,int xo,int xs,__global const double*y,int yo,int ys,__global double*out){if(get_global_id(0)==0){double s=0;for(int i=0;i<n;i++)s+=x[xo+i*xs]*y[yo+i*ys];out[0]=s;}}\n"
			+ "__kernel void blas_nrm2(int n,__global const double*x,int xo,int xs,__global double*out){if(get_global_id(0)==0){double scale=0,sum=1;for(int i=0;i<n;i++){double v=fabs(x[xo+i*xs]);if(v!=0){if(scale<v){double r=scale/v;sum=1+sum*r*r;scale=v;}else{double r=v/scale;sum+=r*r;}}}out[0]=scale==0?0:scale*sqrt(sum);}}\n"
			+ "__kernel void blas_gemv(int tr,int rows,int cols,double alpha,__global const double*a,__global const double*x,double beta,__global double*y){int i=get_global_id(0),output=tr?cols:rows,input=tr?rows:cols;if(i<output){double s=0;for(int j=0;j<input;j++)s+=(tr?a[j*cols+i]:a[i*cols+j])*x[j];y[i]=alpha*s+beta*y[i];}}\n"
			+ "__kernel void blas_gemm(int ta,int tb,int m,int n,int k,double alpha,__global const double*a,__global const double*b,double beta,__global double*c){int row=get_global_id(1),col=get_global_id(0);if(row<m&&col<n){double s=0;for(int q=0;q<k;q++)s+=(ta?a[q*m+row]:a[row*k+q])*(tb?b[col*k+q]:b[q*n+col]);int z=row*n+col;c[z]=alpha*s+beta*c[z];}}\n"
			+ "__kernel void blas_syrk(int tr,int n,int k,double alpha,__global const double*a,double beta,__global double*c){int row=get_global_id(1),col=get_global_id(0);if(row<n&&col<n){double s=0;for(int q=0;q<k;q++)s+=(tr?a[q*n+row]:a[row*k+q])*(tr?a[q*n+col]:a[col*k+q]);c[row*n+col]=alpha*s+beta*c[row*n+col];}}\n"
			+ "__kernel void blas_trsv(int lower,int tr,int unit,int n,__global const double*a,__global double*x){if(get_global_id(0))return;int effective=tr?!lower:lower;for(int step=0;step<n;step++){int i=effective?step:n-1-step;double v=x[i];if(effective){for(int j=0;j<i;j++)v-=(tr?a[j*n+i]:a[i*n+j])*x[j];}else{for(int j=i+1;j<n;j++)v-=(tr?a[j*n+i]:a[i*n+j])*x[j];}x[i]=unit?v:v/a[i*n+i];}}\n"
			+ "__kernel void blas_trsm(int side,int lower,int tr,int unit,int rows,int cols,double alpha,__global const double*a,__global double*b){int vector=get_global_id(0),count=side?rows:cols,n=side?cols:rows;if(vector>=count)return;if(!side){int effective=tr?!lower:lower;for(int i=0;i<n;i++)b[i*cols+vector]*=alpha;for(int step=0;step<n;step++){int i=effective?step:n-1-step;double v=b[i*cols+vector];if(effective){for(int j=0;j<i;j++)v-=(tr?a[j*n+i]:a[i*n+j])*b[j*cols+vector];}else{for(int j=i+1;j<n;j++)v-=(tr?a[j*n+i]:a[i*n+j])*b[j*cols+vector];}b[i*cols+vector]=unit?v:v/a[i*n+i];}}else{int effective=tr?!lower:lower;for(int j=0;j<n;j++)b[vector*cols+j]*=alpha;for(int step=0;step<n;step++){int j=effective?n-1-step:step;double v=b[vector*cols+j];if(effective){for(int k=j+1;k<n;k++)v-=b[vector*cols+k]*(tr?a[j*n+k]:a[k*n+j]);}else{for(int k=0;k<j;k++)v-=b[vector*cols+k]*(tr?a[j*n+k]:a[k*n+j]);}b[vector*cols+j]=unit?v:v/a[j*n+j];}}}\n"
			+ "__kernel void csr_mv(int rows,double alpha,__global const double*v,__global const int*ci,__global const int*rs,__global const double*x,double beta,__global double*y){int row=get_global_id(0);if(row<rows){double s=0;for(int z=rs[row]-1;z<rs[row+1]-1;z++)s+=v[z]*x[ci[z]-1];y[row]=alpha*s+beta*y[row];}}\n"
			+ "__kernel void csr_mm(int rows,int outcols,double alpha,__global const double*v,__global const int*ci,__global const int*rs,__global const double*b,double beta,__global double*c){int z=get_global_id(0);if(z<rows*outcols){int row=z/outcols,col=z-row*outcols;double s=0;for(int q=rs[row]-1;q<rs[row+1]-1;q++)s+=v[q]*b[(ci[q]-1)*outcols+col];c[z]=alpha*s+beta*c[z];}}\n"
			+ "__kernel void float_blas_axpy(int n,float alpha,__global const float*x,int xo,int xs,__global float*y,int yo,int ys){int i=get_global_id(0);if(i<n)y[yo+i*ys]=alpha*x[xo+i*xs]+y[yo+i*ys];}\n"
			+ "__kernel void float_blas_dot(int n,__global const float*x,int xo,int xs,__global const float*y,int yo,int ys,__global float*out){if(get_global_id(0)==0){float s=0;for(int i=0;i<n;i++)s+=x[xo+i*xs]*y[yo+i*ys];out[0]=s;}}\n"
			+ "__kernel void float_blas_nrm2(int n,__global const float*x,int xo,int xs,__global float*out){if(get_global_id(0)==0){float scale=0,sum=1;for(int i=0;i<n;i++){float v=fabs(x[xo+i*xs]);if(v!=0){if(scale<v){float r=scale/v;sum=1+sum*r*r;scale=v;}else{float r=v/scale;sum+=r*r;}}}out[0]=scale==0?0:scale*sqrt(sum);}}\n"
			+ "__kernel void float_blas_gemv(int tr,int rows,int cols,float alpha,__global const float*a,__global const float*x,float beta,__global float*y){int i=get_global_id(0),output=tr?cols:rows,input=tr?rows:cols;if(i<output){float s=0;for(int j=0;j<input;j++)s+=(tr?a[j*cols+i]:a[i*cols+j])*x[j];y[i]=alpha*s+beta*y[i];}}\n"
			+ "__kernel void float_blas_gemm(int ta,int tb,int m,int n,int k,float alpha,__global const float*a,__global const float*b,float beta,__global float*c){int row=get_global_id(1),col=get_global_id(0);if(row<m&&col<n){float s=0;for(int q=0;q<k;q++)s+=(ta?a[q*m+row]:a[row*k+q])*(tb?b[col*k+q]:b[q*n+col]);int z=row*n+col;c[z]=alpha*s+beta*c[z];}}\n"
			+ "__kernel void float_blas_syrk(int tr,int n,int k,float alpha,__global const float*a,float beta,__global float*c){int row=get_global_id(1),col=get_global_id(0);if(row<n&&col<n){float s=0;for(int q=0;q<k;q++)s+=(tr?a[q*n+row]:a[row*k+q])*(tr?a[q*n+col]:a[col*k+q]);c[row*n+col]=alpha*s+beta*c[row*n+col];}}\n"
			+ "__kernel void float_blas_trsv(int lower,int tr,int unit,int n,__global const float*a,__global float*x){if(get_global_id(0))return;int effective=tr?!lower:lower;for(int step=0;step<n;step++){int i=effective?step:n-1-step;float v=x[i];if(effective){for(int j=0;j<i;j++)v-=(tr?a[j*n+i]:a[i*n+j])*x[j];}else{for(int j=i+1;j<n;j++)v-=(tr?a[j*n+i]:a[i*n+j])*x[j];}x[i]=unit?v:v/a[i*n+i];}}\n"
			+ "__kernel void float_blas_trsm(int side,int lower,int tr,int unit,int rows,int cols,float alpha,__global const float*a,__global float*b){int vector=get_global_id(0),count=side?rows:cols,n=side?cols:rows;if(vector>=count)return;if(!side){int effective=tr?!lower:lower;for(int i=0;i<n;i++)b[i*cols+vector]*=alpha;for(int step=0;step<n;step++){int i=effective?step:n-1-step;float v=b[i*cols+vector];if(effective){for(int j=0;j<i;j++)v-=(tr?a[j*n+i]:a[i*n+j])*b[j*cols+vector];}else{for(int j=i+1;j<n;j++)v-=(tr?a[j*n+i]:a[i*n+j])*b[j*cols+vector];}b[i*cols+vector]=unit?v:v/a[i*n+i];}}else{int effective=tr?!lower:lower;for(int j=0;j<n;j++)b[vector*cols+j]*=alpha;for(int step=0;step<n;step++){int j=effective?n-1-step:step;float v=b[vector*cols+j];if(effective){for(int k=j+1;k<n;k++)v-=b[vector*cols+k]*(tr?a[j*n+k]:a[k*n+j]);}else{for(int k=0;k<j;k++)v-=b[vector*cols+k]*(tr?a[j*n+k]:a[k*n+j]);}b[vector*cols+j]=unit?v:v/a[j*n+j];}}}\n"
			+ "__kernel void float_csr_mv(int rows,float alpha,__global const float*v,__global const int*ci,__global const int*rs,__global const float*x,float beta,__global float*y){int row=get_global_id(0);if(row<rows){float s=0;for(int z=rs[row]-1;z<rs[row+1]-1;z++)s+=v[z]*x[ci[z]-1];y[row]=alpha*s+beta*y[row];}}\n"
			+ "__kernel void float_csr_mm(int rows,int outcols,float alpha,__global const float*v,__global const int*ci,__global const int*rs,__global const float*b,float beta,__global float*c){int z=get_global_id(0);if(z<rows*outcols){int row=z/outcols,col=z-row*outcols;float s=0;for(int q=rs[row]-1;q<rs[row+1]-1;q++)s+=v[q]*b[(ci[q]-1)*outcols+col];c[z]=alpha*s+beta*c[z];}}\n"
			+ sparseSource("double","d") + sparseSource("float","s")
			+ decompositionSource("double","d") + decompositionSource("float","s")
			+ "__kernel void transpose_product(__global const double*x,__global const double*v,__global double*out,int rows,int cols,int batches){int z=get_global_id(0);if(z<cols*batches){int b=z/cols,col=z-b*cols;double s=0;for(int row=0;row<rows;row++)s+=x[row*cols+col]*v[b*rows+row];out[z]=s;}}\n"
			+ "__kernel void logistic_residual(__global const double*x,__global const double*y,__global const double*q,__global double*r,__global double*t,int rows,int dims,int chains){int z=get_global_id(0);if(z<rows*chains){int c=z/rows,i=z-c*rows;double eta=0;for(int d=0;d<dims;d++)eta+=x[i*dims+d]*q[c*dims+d];r[z]=y[i]-logistic(eta);t[z]=y[i]*eta-l1e(eta);}}\n"
			+ "__kernel void logistic_gradient(__global const double*x,__global const double*q,__global const double*r,__global double*g,int rows,int dims,int chains,double prior){int z=get_global_id(0);if(z<dims*chains){int c=z/dims,d=z-c*dims;double s=-prior*q[z];for(int i=0;i<rows;i++)s+=r[c*rows+i]*x[i*dims+d];g[z]=s;}}\n"
			+ "__kernel void logistic_logp(__global const double*t,__global const double*q,__global double*out,int rows,int dims,int chains,double prior){int c=get_global_id(0);if(c<chains){double s=0;for(int i=0;i<rows;i++)s+=t[c*rows+i];for(int d=0;d<dims;d++){double v=q[c*dims+d];s-=0.5*prior*v*v;}out[c]=s;}}\n";

	private cl_context context;
	private cl_command_queue queue;
	private cl_program program;
	private ComputeCapabilities capabilities;
	private ComputeDeviceInfo deviceInfo;
	private Throwable unavailableCause;

	/** Detects the first FP64 GPU and makes this instance unavailable when none can initialize. */
	public OpenClComputeBackend() {
		try { initialize(); } catch (Throwable error) { unavailableCause = error; close(); }
	}
	@Override public String id() { return "opencl"; }
	@Override public boolean available() { return unavailableCause == null && context != null; }
	@Override public ComputeCapabilities capabilities() { ensureAvailable(); return capabilities; }
	@Override public ComputeDeviceInfo deviceInfo() { ensureAvailable(); return deviceInfo; }
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
	@Override public synchronized void daxpy(int count, double alpha, double[] x,
			int xOffset, int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return; ensureAvailable(); cl_mem dx = input(x), dy = inputReadWrite(y);
		cl_kernel operation = kernel("blas_axpy");
		try {
			arg(operation, 0, count); arg(operation, 1, alpha); arg(operation, 2, dx);
			arg(operation, 3, xOffset); arg(operation, 4, xStride); arg(operation, 5, dy);
			arg(operation, 6, yOffset); arg(operation, 7, yStride); run1d(operation, count);
			double[] updated = read(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized double ddot(int count, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0; ensureAvailable(); cl_mem dx = input(x), dy = input(y), out = output(1);
		cl_kernel operation = kernel("blas_dot");
		try {
			arg(operation, 0, count); arg(operation, 1, dx); arg(operation, 2, xOffset); arg(operation, 3, xStride);
			arg(operation, 4, dy); arg(operation, 5, yOffset); arg(operation, 6, yStride); arg(operation, 7, out);
			run1d(operation, 1); return read(out, 1)[0];
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(dy); clReleaseMemObject(out); }
	}
	@Override public synchronized double dnrm2(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0;
		ensureAvailable(); cl_mem dx = input(x), out = output(1); cl_kernel operation = kernel("blas_nrm2");
		try {
			arg(operation, 0, count); arg(operation, 1, dx); arg(operation, 2, offset); arg(operation, 3, stride);
			arg(operation, 4, out); run1d(operation, 1); return read(out, 1)[0];
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(out); }
	}
	@Override public synchronized void dgemv(MatrixTranspose transpose, int rows, int columns,
			double alpha, double[] matrix, double[] x, double beta, double[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable();
		cl_mem a = input(matrix), dx = input(x), dy = inputReadWrite(y); cl_kernel operation = kernel("blas_gemv");
		try {
			arg(operation, 0, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0); arg(operation, 1, rows);
			arg(operation, 2, columns); arg(operation, 3, alpha); arg(operation, 4, a); arg(operation, 5, dx);
			arg(operation, 6, beta); arg(operation, 7, dy); run1d(operation, y.length);
			double[] updated = read(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left, double[] right,
			double beta, double[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result); ensureAvailable();
		cl_mem a = input(left), b = input(right), c = inputReadWrite(result); cl_kernel operation = kernel("blas_gemm");
		try {
			arg(operation, 0, leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 1, rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 2, rows); arg(operation, 3, columns); arg(operation, 4, shared); arg(operation, 5, alpha);
			arg(operation, 6, a); arg(operation, 7, b); arg(operation, 8, beta); arg(operation, 9, c);
			check(clEnqueueNDRangeKernel(queue, operation, 2, null,
					new long[] {round16(columns), round16(rows)}, new long[] {16, 16}, 0, null, null));
			double[] updated = read(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(c); }
	}
	@Override public synchronized void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, double beta, double[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result); ensureAvailable();
		cl_mem a = input(matrix), c = inputReadWrite(result); cl_kernel operation = kernel("blas_syrk");
		try {
			arg(operation, 0, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0); arg(operation, 1, dimension);
			arg(operation, 2, shared); arg(operation, 3, alpha); arg(operation, 4, a); arg(operation, 5, beta); arg(operation, 6, c);
			check(clEnqueueNDRangeKernel(queue, operation, 2, null, new long[] {round16(dimension), round16(dimension)},
					new long[] {16, 16}, 0, null, null));
			double[] updated = read(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(c); }
	}
	@Override public synchronized void dtrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector); ensureAvailable();
		cl_mem a = input(matrix), x = inputReadWrite(vector); cl_kernel operation = kernel("blas_trsv");
		try {
			arg(operation, 0, triangle == MatrixTriangle.LOWER ? 1 : 0);
			arg(operation, 1, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 2, diagonal == MatrixDiagonal.UNIT ? 1 : 0); arg(operation, 3, dimension);
			arg(operation, 4, a); arg(operation, 5, x); run1d(operation, 1);
			double[] updated = read(x, vector.length); System.arraycopy(updated, 0, vector, 0, vector.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(x); }
	}
	@Override public synchronized void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, double[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right); ensureAvailable();
		cl_mem a = input(matrix), b = inputReadWrite(right); cl_kernel operation = kernel("blas_trsm");
		try {
			arg(operation, 0, side == MatrixSide.RIGHT ? 1 : 0); arg(operation, 1, triangle == MatrixTriangle.LOWER ? 1 : 0);
			arg(operation, 2, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 3, diagonal == MatrixDiagonal.UNIT ? 1 : 0); arg(operation, 4, rows); arg(operation, 5, columns);
			arg(operation, 6, alpha); arg(operation, 7, a); arg(operation, 8, b);
			run1d(operation, side == MatrixSide.LEFT ? columns : rows);
			double[] updated = read(b, right.length); System.arraycopy(updated, 0, right, 0, right.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(b); }
	}
	@Override public synchronized void dcsrmv(double alpha, CsrMatrix matrix, double[] x,
			double beta, double[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		cl_mem values = input(matrix.values());
		cl_mem indices = input(matrix.columnIndices()), starts = input(matrix.rowStarts());
		cl_mem dx = input(x), dy = inputReadWrite(y); cl_kernel operation = kernel("csr_mv");
		try {
			arg(operation, 0, matrix.rows()); arg(operation, 1, alpha); arg(operation, 2, values);
			arg(operation, 3, indices); arg(operation, 4, starts); arg(operation, 5, dx); arg(operation, 6, beta); arg(operation, 7, dy);
			run1d(operation, matrix.rows()); double[] updated = read(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(values); clReleaseMemObject(indices);
			clReleaseMemObject(starts); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		cl_mem values = input(matrix.values());
		cl_mem indices = input(matrix.columnIndices()), starts = input(matrix.rowStarts());
		cl_mem b = input(right), c = inputReadWrite(result); cl_kernel operation = kernel("csr_mm");
		try {
			arg(operation, 0, matrix.rows()); arg(operation, 1, rightColumns); arg(operation, 2, alpha);
			arg(operation, 3, values); arg(operation, 4, indices); arg(operation, 5, starts);
			arg(operation, 6, b); arg(operation, 7, beta); arg(operation, 8, c); run1d(operation, result.length);
			double[] updated = read(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(values); clReleaseMemObject(indices);
			clReleaseMemObject(starts); clReleaseMemObject(b); clReleaseMemObject(c); }
	}
	@Override public synchronized void saxpy(int count, float alpha, float[] x,
			int xOffset, int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return; ensureAvailable(); cl_mem dx = input(x), dy = inputReadWrite(y);
		cl_kernel operation = kernel("float_blas_axpy");
		try {
			arg(operation, 0, count); arg(operation, 1, alpha); arg(operation, 2, dx);
			arg(operation, 3, xOffset); arg(operation, 4, xStride); arg(operation, 5, dy);
			arg(operation, 6, yOffset); arg(operation, 7, yStride); run1d(operation, count);
			float[] updated = readFloats(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized float sdot(int count, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0f; ensureAvailable();
		cl_mem dx = input(x), dy = input(y), out = outputFloats(1); cl_kernel operation = kernel("float_blas_dot");
		try {
			arg(operation, 0, count); arg(operation, 1, dx); arg(operation, 2, xOffset); arg(operation, 3, xStride);
			arg(operation, 4, dy); arg(operation, 5, yOffset); arg(operation, 6, yStride); arg(operation, 7, out);
			run1d(operation, 1); return readFloats(out, 1)[0];
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(dy); clReleaseMemObject(out); }
	}
	@Override public synchronized float snrm2(int count, float[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0f;
		ensureAvailable(); cl_mem dx = input(x), out = outputFloats(1); cl_kernel operation = kernel("float_blas_nrm2");
		try {
			arg(operation, 0, count); arg(operation, 1, dx); arg(operation, 2, offset); arg(operation, 3, stride);
			arg(operation, 4, out); run1d(operation, 1); return readFloats(out, 1)[0];
		} finally { clReleaseKernel(operation); clReleaseMemObject(dx); clReleaseMemObject(out); }
	}
	@Override public synchronized void sgemv(MatrixTranspose transpose, int rows, int columns,
			float alpha, float[] matrix, float[] x, float beta, float[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable();
		cl_mem a = input(matrix), dx = input(x), dy = inputReadWrite(y); cl_kernel operation = kernel("float_blas_gemv");
		try {
			arg(operation, 0, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0); arg(operation, 1, rows);
			arg(operation, 2, columns); arg(operation, 3, alpha); arg(operation, 4, a); arg(operation, 5, dx);
			arg(operation, 6, beta); arg(operation, 7, dy); run1d(operation, y.length);
			float[] updated = readFloats(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left, float[] right,
			float beta, float[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result); ensureAvailable();
		cl_mem a = input(left), b = input(right), c = inputReadWrite(result); cl_kernel operation = kernel("float_blas_gemm");
		try {
			arg(operation, 0, leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 1, rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 2, rows); arg(operation, 3, columns); arg(operation, 4, shared); arg(operation, 5, alpha);
			arg(operation, 6, a); arg(operation, 7, b); arg(operation, 8, beta); arg(operation, 9, c);
			check(clEnqueueNDRangeKernel(queue, operation, 2, null,
					new long[] {round16(columns), round16(rows)}, new long[] {16, 16}, 0, null, null));
			float[] updated = readFloats(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(c); }
	}
	@Override public synchronized void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, float beta, float[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result); ensureAvailable();
		cl_mem a = input(matrix), c = inputReadWrite(result); cl_kernel operation = kernel("float_blas_syrk");
		try {
			arg(operation, 0, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0); arg(operation, 1, dimension);
			arg(operation, 2, shared); arg(operation, 3, alpha); arg(operation, 4, a); arg(operation, 5, beta); arg(operation, 6, c);
			check(clEnqueueNDRangeKernel(queue, operation, 2, null, new long[] {round16(dimension), round16(dimension)},
					new long[] {16, 16}, 0, null, null));
			float[] updated = readFloats(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(c); }
	}
	@Override public synchronized void strsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector); ensureAvailable();
		cl_mem a = input(matrix), x = inputReadWrite(vector); cl_kernel operation = kernel("float_blas_trsv");
		try {
			arg(operation, 0, triangle == MatrixTriangle.LOWER ? 1 : 0);
			arg(operation, 1, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 2, diagonal == MatrixDiagonal.UNIT ? 1 : 0); arg(operation, 3, dimension);
			arg(operation, 4, a); arg(operation, 5, x); run1d(operation, 1);
			float[] updated = readFloats(x, vector.length); System.arraycopy(updated, 0, vector, 0, vector.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(x); }
	}
	@Override public synchronized void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, float[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right); ensureAvailable();
		cl_mem a = input(matrix), b = inputReadWrite(right); cl_kernel operation = kernel("float_blas_trsm");
		try {
			arg(operation, 0, side == MatrixSide.RIGHT ? 1 : 0); arg(operation, 1, triangle == MatrixTriangle.LOWER ? 1 : 0);
			arg(operation, 2, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0);
			arg(operation, 3, diagonal == MatrixDiagonal.UNIT ? 1 : 0); arg(operation, 4, rows); arg(operation, 5, columns);
			arg(operation, 6, alpha); arg(operation, 7, a); arg(operation, 8, b);
			run1d(operation, side == MatrixSide.LEFT ? columns : rows);
			float[] updated = readFloats(b, right.length); System.arraycopy(updated, 0, right, 0, right.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(a); clReleaseMemObject(b); }
	}
	@Override public synchronized void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x,
			float beta, float[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		cl_mem values = input(matrix.values());
		cl_mem indices = input(matrix.columnIndices()), starts = input(matrix.rowStarts());
		cl_mem dx = input(x), dy = inputReadWrite(y); cl_kernel operation = kernel("float_csr_mv");
		try {
			arg(operation, 0, matrix.rows()); arg(operation, 1, alpha); arg(operation, 2, values);
			arg(operation, 3, indices); arg(operation, 4, starts); arg(operation, 5, dx); arg(operation, 6, beta); arg(operation, 7, dy);
			run1d(operation, matrix.rows()); float[] updated = readFloats(dy, y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(values); clReleaseMemObject(indices);
			clReleaseMemObject(starts); clReleaseMemObject(dx); clReleaseMemObject(dy); }
	}
	@Override public synchronized void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		cl_mem values = input(matrix.values());
		cl_mem indices = input(matrix.columnIndices()), starts = input(matrix.rowStarts());
		cl_mem b = input(right), c = inputReadWrite(result); cl_kernel operation = kernel("float_csr_mm");
		try {
			arg(operation, 0, matrix.rows()); arg(operation, 1, rightColumns); arg(operation, 2, alpha);
			arg(operation, 3, values); arg(operation, 4, indices); arg(operation, 5, starts);
			arg(operation, 6, b); arg(operation, 7, beta); arg(operation, 8, c); run1d(operation, result.length);
			float[] updated = readFloats(c, result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { clReleaseKernel(operation); clReleaseMemObject(values); clReleaseMemObject(indices);
			clReleaseMemObject(starts); clReleaseMemObject(b); clReleaseMemObject(c); }
	}
	@Override public synchronized CholeskyFactor dpotrf(double[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);ensureAvailable();cl_mem a=input(matrix),lower=output(matrix.length),info=outputInts(1);cl_kernel operation=kernel("decomp_dpotrf");try{arg(operation,0,a);arg(operation,1,lower);arg(operation,2,dimension);arg(operation,3,info);run1d(operation,1);int status=readInts(info,1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);return new CholeskyFactor(dimension,read(lower,matrix.length));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(lower);clReleaseMemObject(a);}}
	@Override public synchronized PivotedQrFactor dgeqp3(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);cl_mem qr=inputReadWrite(matrix),tau=output(count),pivot=outputInts(columns);cl_kernel operation=kernel("decomp_dgeqp3");try{arg(operation,0,qr);arg(operation,1,tau);arg(operation,2,pivot);arg(operation,3,rows);arg(operation,4,columns);run1d(operation,1);return new PivotedQrFactor(rows,columns,read(qr,matrix.length),read(tau,count),readInts(pivot,columns));}finally{clReleaseKernel(operation);clReleaseMemObject(pivot);clReleaseMemObject(tau);clReleaseMemObject(qr);}}
	@Override public synchronized SymmetricEigenDecomposition dsyev(double[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();cl_mem work=inputReadWrite(matrix),values=output(dimension),vectors=output(matrix.length),info=outputInts(1);cl_kernel operation=kernel("decomp_dsyev");try{arg(operation,0,work);arg(operation,1,values);arg(operation,2,vectors);arg(operation,3,dimension);arg(operation,4,16*Math.ulp(1.0));arg(operation,5,info);run1d(operation,1);if(readInts(info,1)[0]!=0)throw new IllegalStateException("OpenCL symmetric eigendecomposition did not converge");return new SymmetricEigenDecomposition(dimension,read(values,dimension),read(vectors,matrix.length));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(vectors);clReleaseMemObject(values);clReleaseMemObject(work);}}
	@Override public synchronized SingularValueDecomposition dgesvd(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);cl_mem a=input(matrix),work=output(matrix.length),u=output(rows*count),singular=output(count),vt=output(count*columns),info=outputInts(1);cl_kernel operation=kernel("decomp_dgesvd");try{arg(operation,0,a);arg(operation,1,work);arg(operation,2,u);arg(operation,3,singular);arg(operation,4,vt);arg(operation,5,rows);arg(operation,6,columns);arg(operation,7,16*Math.ulp(1.0));arg(operation,8,info);run1d(operation,1);if(readInts(info,1)[0]!=0)throw new IllegalStateException("OpenCL SVD did not converge");return new SingularValueDecomposition(rows,columns,read(singular,count),read(u,rows*count),read(vt,count*columns));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(vt);clReleaseMemObject(singular);clReleaseMemObject(u);clReleaseMemObject(work);clReleaseMemObject(a);}}
	@Override public synchronized FloatCholeskyFactor spotrf(float[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);ensureAvailable();cl_mem a=input(matrix),lower=outputFloats(matrix.length),info=outputInts(1);cl_kernel operation=kernel("decomp_spotrf");try{arg(operation,0,a);arg(operation,1,lower);arg(operation,2,dimension);arg(operation,3,info);run1d(operation,1);int status=readInts(info,1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);return new FloatCholeskyFactor(dimension,readFloats(lower,matrix.length));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(lower);clReleaseMemObject(a);}}
	@Override public synchronized FloatPivotedQrFactor sgeqp3(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);cl_mem qr=inputReadWrite(matrix),tau=outputFloats(count),pivot=outputInts(columns);cl_kernel operation=kernel("decomp_sgeqp3");try{arg(operation,0,qr);arg(operation,1,tau);arg(operation,2,pivot);arg(operation,3,rows);arg(operation,4,columns);run1d(operation,1);return new FloatPivotedQrFactor(rows,columns,readFloats(qr,matrix.length),readFloats(tau,count),readInts(pivot,columns));}finally{clReleaseKernel(operation);clReleaseMemObject(pivot);clReleaseMemObject(tau);clReleaseMemObject(qr);}}
	@Override public synchronized FloatSymmetricEigenDecomposition ssyev(float[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();cl_mem work=inputReadWrite(matrix),values=outputFloats(dimension),vectors=outputFloats(matrix.length),info=outputInts(1);cl_kernel operation=kernel("decomp_ssyev");try{arg(operation,0,work);arg(operation,1,values);arg(operation,2,vectors);arg(operation,3,dimension);arg(operation,4,16*Math.ulp(1.0f));arg(operation,5,info);run1d(operation,1);if(readInts(info,1)[0]!=0)throw new IllegalStateException("OpenCL FP32 symmetric eigendecomposition did not converge");return new FloatSymmetricEigenDecomposition(dimension,readFloats(values,dimension),readFloats(vectors,matrix.length));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(vectors);clReleaseMemObject(values);clReleaseMemObject(work);}}
	@Override public synchronized FloatSingularValueDecomposition sgesvd(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);cl_mem a=input(matrix),work=outputFloats(matrix.length),u=outputFloats(rows*count),singular=outputFloats(count),vt=outputFloats(count*columns),info=outputInts(1);cl_kernel operation=kernel("decomp_sgesvd");try{arg(operation,0,a);arg(operation,1,work);arg(operation,2,u);arg(operation,3,singular);arg(operation,4,vt);arg(operation,5,rows);arg(operation,6,columns);arg(operation,7,16*Math.ulp(1.0f));arg(operation,8,info);run1d(operation,1);if(readInts(info,1)[0]!=0)throw new IllegalStateException("OpenCL FP32 SVD did not converge");return new FloatSingularValueDecomposition(rows,columns,readFloats(singular,count),readFloats(u,rows*count),readFloats(vt,count*columns));}finally{clReleaseKernel(operation);clReleaseMemObject(info);clReleaseMemObject(vt);clReleaseMemObject(singular);clReleaseMemObject(u);clReleaseMemObject(work);clReleaseMemObject(a);}}
	@Override public synchronized LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		PreparedLogisticRegression prepared = prepareLogisticRegression(design, outcomes);
		try { return prepared.evaluate(states, priorPrecision); } finally { prepared.close(); }
	}
	@Override public synchronized PreparedTransposeProduct prepareTransposeProduct(double[][] matrix) {
		int[] matrixShape = shape(matrix); ensureAvailable();
		return new PreparedTranspose(matrix, matrixShape[0], matrixShape[1]);
	}
	@Override public synchronized PreparedDenseMatrix prepareDge(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();return new PreparedDoubleDense(matrix,rows,columns);}
	@Override public synchronized PreparedFloatDenseMatrix prepareSge(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();return new PreparedFloatDense(matrix,rows,columns);}
	private final class PreparedDoubleDense implements PreparedDenseMatrix{
		private final int rows,columns;private cl_mem matrix;
		PreparedDoubleDense(double[]source,int rows,int columns){this.rows=rows;this.columns=columns;matrix=input(source);}
		public int rows(){checkOpen();return rows;}public int columns(){checkOpen();return columns;}
		public void multiply(MatrixTranspose transpose,double alpha,double[]right,int rightColumns,double beta,double[]result){if(transpose==null||rightColumns<1)throw new IllegalArgumentException("prepared OpenCL dense dimensions do not conform");int output=transpose==MatrixTranspose.NONE?rows:columns,shared=transpose==MatrixTranspose.NONE?columns:rows;if(right==null||right.length!=shared*rightColumns||result==null||result.length!=output*rightColumns)throw new IllegalArgumentException("prepared OpenCL dense dimensions do not conform");synchronized(OpenClComputeBackend.this){checkOpen();cl_mem b=input(right),c=inputReadWrite(result);cl_kernel operation=kernel("blas_gemm");try{arg(operation,0,transpose==MatrixTranspose.TRANSPOSE?1:0);arg(operation,1,0);arg(operation,2,output);arg(operation,3,rightColumns);arg(operation,4,shared);arg(operation,5,alpha);arg(operation,6,matrix);arg(operation,7,b);arg(operation,8,beta);arg(operation,9,c);check(clEnqueueNDRangeKernel(queue,operation,2,null,new long[]{round16(rightColumns),round16(output)},new long[]{16,16},0,null,null));double[]updated=read(c,result.length);System.arraycopy(updated,0,result,0,result.length);}finally{clReleaseKernel(operation);clReleaseMemObject(b);clReleaseMemObject(c);}}}
		public void close(){synchronized(OpenClComputeBackend.this){if(matrix!=null){clReleaseMemObject(matrix);matrix=null;}}}private void checkOpen(){if(matrix==null)throw new IllegalStateException("prepared OpenCL dense matrix is closed");}
	}
	private final class PreparedFloatDense implements PreparedFloatDenseMatrix{
		private final int rows,columns;private cl_mem matrix;
		PreparedFloatDense(float[]source,int rows,int columns){this.rows=rows;this.columns=columns;matrix=input(source);}
		public int rows(){checkOpen();return rows;}public int columns(){checkOpen();return columns;}
		public void multiply(MatrixTranspose transpose,float alpha,float[]right,int rightColumns,float beta,float[]result){if(transpose==null||rightColumns<1)throw new IllegalArgumentException("prepared OpenCL FP32 dense dimensions do not conform");int output=transpose==MatrixTranspose.NONE?rows:columns,shared=transpose==MatrixTranspose.NONE?columns:rows;if(right==null||right.length!=shared*rightColumns||result==null||result.length!=output*rightColumns)throw new IllegalArgumentException("prepared OpenCL FP32 dense dimensions do not conform");synchronized(OpenClComputeBackend.this){checkOpen();cl_mem b=input(right),c=inputReadWrite(result);cl_kernel operation=kernel("float_blas_gemm");try{arg(operation,0,transpose==MatrixTranspose.TRANSPOSE?1:0);arg(operation,1,0);arg(operation,2,output);arg(operation,3,rightColumns);arg(operation,4,shared);arg(operation,5,alpha);arg(operation,6,matrix);arg(operation,7,b);arg(operation,8,beta);arg(operation,9,c);check(clEnqueueNDRangeKernel(queue,operation,2,null,new long[]{round16(rightColumns),round16(output)},new long[]{16,16},0,null,null));float[]updated=readFloats(c,result.length);System.arraycopy(updated,0,result,0,result.length);}finally{clReleaseKernel(operation);clReleaseMemObject(b);clReleaseMemObject(c);}}}
		public void close(){synchronized(OpenClComputeBackend.this){if(matrix!=null){clReleaseMemObject(matrix);matrix=null;}}}private void checkOpen(){if(matrix==null)throw new IllegalStateException("prepared OpenCL FP32 dense matrix is closed");}
	}
	@Override public PreparedSparseCholesky prepareDcsrpotrf(CsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return new PreparedDoubleSparseFactor(matrix, triangle, ordering);
	}
	@Override public PreparedFloatSparseCholesky prepareScsrpotrf(FloatCsrMatrix matrix,
			MatrixTriangle triangle, SparseOrdering ordering) {
		return new PreparedFloatSparseFactor(matrix, triangle, ordering);
	}
	private final class PreparedDoubleSparseFactor implements PreparedSparseCholesky {
		private final SparseCholeskyPlan plan;
		private cl_mem factor, candidate, indices, starts, permutation, info, determinant;
		private double logDeterminant; private boolean closed;
		PreparedDoubleSparseFactor(CsrMatrix matrix, MatrixTriangle triangle, SparseOrdering ordering) {
			plan = SparseCholeskyPlan.analyze(matrix, triangle, ordering);
			synchronized (OpenClComputeBackend.this) {
				ensureAvailable(); int count = plan.factorNonzeroCount(); factor = output(count); candidate = output(count);
				indices = input(plan.factorColumnIndices()); starts = input(plan.factorRowStarts());
				permutation = input(plan.permutation()); info = outputInts(1); determinant = output(1);
				try { factor(plan.factorValues(matrix)); } catch (RuntimeException error) { close(); throw error; }
			}
		}
		@Override public int dimension() { checkOpen(); return plan.dimension(); }
		@Override public int structuralNonzeroCount() { checkOpen(); return plan.structuralNonzeroCount(); }
		@Override public int factorNonzeroCount() { checkOpen(); return plan.factorNonzeroCount(); }
		@Override public int[] permutation() { checkOpen(); return plan.permutation(); }
		@Override public double logDeterminant() { checkOpen(); return logDeterminant; }
		@Override public void refactor(CsrMatrix matrix) { double[] values = plan.factorValues(matrix);
			synchronized (OpenClComputeBackend.this) { checkOpen(); factor(values); } }
		private void factor(double[] values) {
			cl_mem source = input(values); cl_kernel operation = kernel("sparse_dpotrf");
			try { arg(operation,0,source);arg(operation,1,candidate);arg(operation,2,indices);arg(operation,3,starts);
				arg(operation,4,plan.dimension());arg(operation,5,info);arg(operation,6,determinant);run1d(operation,1);
				int status=readInts(info,1)[0];if(status!=0)throw new IllegalArgumentException(
						"sparse matrix is not positive definite at permuted minor "+status);
				logDeterminant=read(determinant,1)[0];cl_mem previous=factor;factor=candidate;candidate=previous;
			} finally { clReleaseKernel(operation);clReleaseMemObject(source); }
		}
		@Override public void solveInPlace(double[] right,int columns) {
			if(columns<1||right==null||right.length!=plan.dimension()*columns)
				throw new IllegalArgumentException("invalid OpenCL sparse right side");
			synchronized(OpenClComputeBackend.this){checkOpen();cl_mem source=input(right),work=output(right.length),result=output(right.length);
				cl_kernel operation=kernel("sparse_dsolve");try{arg(operation,0,factor);arg(operation,1,indices);arg(operation,2,starts);
					arg(operation,3,permutation);arg(operation,4,source);arg(operation,5,work);arg(operation,6,result);
					arg(operation,7,plan.dimension());arg(operation,8,columns);run1d(operation,columns);
					double[]updated=read(result,right.length);System.arraycopy(updated,0,right,0,right.length);
				}finally{clReleaseKernel(operation);clReleaseMemObject(result);clReleaseMemObject(work);clReleaseMemObject(source);}}
		}
		@Override public void close(){synchronized(OpenClComputeBackend.this){if(!closed){release(factor);release(candidate);
			release(indices);release(starts);release(permutation);release(info);release(determinant);
			factor=candidate=indices=starts=permutation=info=determinant=null;closed=true;}}}
		private void checkOpen(){if(closed||factor==null)throw new IllegalStateException("prepared OpenCL sparse Cholesky is closed");}
	}
	private final class PreparedFloatSparseFactor implements PreparedFloatSparseCholesky {
		private final SparseCholeskyPlan plan;
		private cl_mem factor, candidate, indices, starts, permutation, info, determinant;
		private float logDeterminant; private boolean closed;
		PreparedFloatSparseFactor(FloatCsrMatrix matrix, MatrixTriangle triangle, SparseOrdering ordering) {
			plan = SparseCholeskyPlan.analyze(matrix, triangle, ordering);
			synchronized (OpenClComputeBackend.this) {
				ensureAvailable(); int count = plan.factorNonzeroCount(); factor = outputFloats(count); candidate = outputFloats(count);
				indices = input(plan.factorColumnIndices()); starts = input(plan.factorRowStarts());
				permutation = input(plan.permutation()); info = outputInts(1); determinant = outputFloats(1);
				try { factor(plan.factorValues(matrix)); } catch (RuntimeException error) { close(); throw error; }
			}
		}
		@Override public int dimension() { checkOpen(); return plan.dimension(); }
		@Override public int structuralNonzeroCount() { checkOpen(); return plan.structuralNonzeroCount(); }
		@Override public int factorNonzeroCount() { checkOpen(); return plan.factorNonzeroCount(); }
		@Override public int[] permutation() { checkOpen(); return plan.permutation(); }
		@Override public float logDeterminant() { checkOpen(); return logDeterminant; }
		@Override public void refactor(FloatCsrMatrix matrix) { float[] values = plan.factorValues(matrix);
			synchronized (OpenClComputeBackend.this) { checkOpen(); factor(values); } }
		private void factor(float[] values) {
			cl_mem source = input(values); cl_kernel operation = kernel("sparse_spotrf");
			try { arg(operation,0,source);arg(operation,1,candidate);arg(operation,2,indices);arg(operation,3,starts);
				arg(operation,4,plan.dimension());arg(operation,5,info);arg(operation,6,determinant);run1d(operation,1);
				int status=readInts(info,1)[0];if(status!=0)throw new IllegalArgumentException(
						"FP32 sparse matrix is not positive definite at permuted minor "+status);
				logDeterminant=readFloats(determinant,1)[0];cl_mem previous=factor;factor=candidate;candidate=previous;
			} finally { clReleaseKernel(operation);clReleaseMemObject(source); }
		}
		@Override public void solveInPlace(float[] right,int columns) {
			if(columns<1||right==null||right.length!=plan.dimension()*columns)
				throw new IllegalArgumentException("invalid OpenCL FP32 sparse right side");
			synchronized(OpenClComputeBackend.this){checkOpen();cl_mem source=input(right),work=outputFloats(right.length),result=outputFloats(right.length);
				cl_kernel operation=kernel("sparse_ssolve");try{arg(operation,0,factor);arg(operation,1,indices);arg(operation,2,starts);
					arg(operation,3,permutation);arg(operation,4,source);arg(operation,5,work);arg(operation,6,result);
					arg(operation,7,plan.dimension());arg(operation,8,columns);run1d(operation,columns);
					float[]updated=readFloats(result,right.length);System.arraycopy(updated,0,right,0,right.length);
				}finally{clReleaseKernel(operation);clReleaseMemObject(result);clReleaseMemObject(work);clReleaseMemObject(source);}}
		}
		@Override public void close(){synchronized(OpenClComputeBackend.this){if(!closed){release(factor);release(candidate);
			release(indices);release(starts);release(permutation);release(info);release(determinant);
			factor=candidate=indices=starts=permutation=info=determinant=null;closed=true;}}}
		private void checkOpen(){if(closed||factor==null)throw new IllegalStateException("prepared OpenCL FP32 sparse Cholesky is closed");}
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
				extensions.contains("cl_khr_fp64") || extensions.contains("cl_amd_fp64"), true,
				memory[0], true, true, true, false, true, true, true, true);
		if (!capabilities.doublePrecision()) throw new IllegalStateException("OpenCL device lacks FP64");
		long[] vendorId = new long[1]; int[] numericVendor = new int[1];
		check(clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, Sizeof.cl_uint, Pointer.to(numericVendor), null));
		vendorId[0] = numericVendor[0] & 0xffffffffL;
		Package pkg = getClass().getPackage(); String backendVersion = pkg == null
				|| pkg.getImplementationVersion() == null ? "development" : pkg.getImplementationVersion();
		deviceInfo = new ComputeDeviceInfo(id(), backendVersion, ComputeApi.OPENCL,
				deviceString(device, CL_DEVICE_VERSION), deviceString(device, CL_DRIVER_VERSION),
				deviceString(device, CL_DEVICE_VENDOR), deviceString(device, CL_DEVICE_NAME),
				System.getProperty("os.arch", "unknown"), Long.toHexString(vendorId[0]), memory[0]);
	}
	private cl_kernel kernel(String name) { int[] status = new int[1]; cl_kernel result = clCreateKernel(program, name, status); check(status[0]); return result; }
	private cl_mem input(double[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_double, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem input(float[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_float, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem input(int[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_int, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem inputReadWrite(double[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_double, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem inputReadWrite(float[] values) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (long) values.length * Sizeof.cl_float, Pointer.to(values), status); check(status[0]); return result; }
	private cl_mem output(int count) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE,
			(long) count * Sizeof.cl_double, null, status); check(status[0]); return result; }
	private cl_mem outputFloats(int count) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE,
			(long) count * Sizeof.cl_float, null, status); check(status[0]); return result; }
	private cl_mem outputInts(int count) { int[] status = new int[1]; cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE,
			(long) count * Sizeof.cl_int, null, status); check(status[0]); return result; }
	private double[] read(cl_mem source, int count) { double[] result = new double[count]; check(clEnqueueReadBuffer(queue,
			source, CL_TRUE, 0, (long) count * Sizeof.cl_double, Pointer.to(result), 0, null, null)); return result; }
	private float[] readFloats(cl_mem source, int count) { float[] result = new float[count]; check(clEnqueueReadBuffer(queue,
			source, CL_TRUE, 0, (long) count * Sizeof.cl_float, Pointer.to(result), 0, null, null)); return result; }
	private int[] readInts(cl_mem source, int count) { int[] result = new int[count]; check(clEnqueueReadBuffer(queue,
			source, CL_TRUE, 0, (long) count * Sizeof.cl_int, Pointer.to(result), 0, null, null)); return result; }
	private static void release(cl_mem memory) { if (memory != null) clReleaseMemObject(memory); }
	private void run1d(cl_kernel kernel, int count) { check(clEnqueueNDRangeKernel(queue, kernel, 1, null,
			new long[] {round(count)}, new long[] {LOCAL}, 0, null, null)); }
	private static void arg(cl_kernel kernel, int index, cl_mem value) { check(clSetKernelArg(kernel, index, Sizeof.cl_mem, Pointer.to(value))); }
	private static void arg(cl_kernel kernel, int index, int value) { check(clSetKernelArg(kernel, index, Sizeof.cl_int, Pointer.to(new int[] {value}))); }
	private static void arg(cl_kernel kernel, int index, double value) { check(clSetKernelArg(kernel, index, Sizeof.cl_double, Pointer.to(new double[] {value}))); }
	private static void arg(cl_kernel kernel, int index, float value) { check(clSetKernelArg(kernel, index, Sizeof.cl_float, Pointer.to(new float[] {value}))); }
	@SuppressWarnings("deprecation")
	private static cl_command_queue createQueue(cl_context context, cl_device_id device, int[] status) {
		cl_command_queue result = null;
		try { result = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), status); }
		catch (LinkageError unavailableOnOpenCl12) { status[0] = CL_INVALID_OPERATION; }
		if (status[0] != CL_SUCCESS || result == null) result = clCreateCommandQueue(context, device, 0, status);
		return result;
	}
	private static long round(int count) { return ((count + LOCAL - 1L) / LOCAL) * LOCAL; }
	private static long round16(int count) { return ((count + 15L) / 16L) * 16L; }
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
		if (transpose == null || rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns
				|| x == null || x.length != (transpose == MatrixTranspose.NONE ? columns : rows)
				|| y == null || y.length != (transpose == MatrixTranspose.NONE ? rows : columns))
			throw new IllegalArgumentException("GEMV dimensions do not conform");
	}
	private static void checkGemv(MatrixTranspose transpose, int rows, int columns,
			float[] matrix, float[] x, float[] y) {
		if (transpose == null || rows < 1 || columns < 1 || matrix == null || matrix.length != rows * columns
				|| x == null || x.length != (transpose == MatrixTranspose.NONE ? columns : rows)
				|| y == null || y.length != (transpose == MatrixTranspose.NONE ? rows : columns))
			throw new IllegalArgumentException("GEMV dimensions do not conform");
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
	private static void checkSyrk(MatrixTranspose transpose, int dimension, int shared,
			double[] matrix, double[] result) {
		if (transpose == null || dimension < 1 || shared < 1 || matrix == null
				|| matrix.length != dimension * shared || result == null
				|| result.length != dimension * dimension)
			throw new IllegalArgumentException("SYRK dimensions do not conform");
	}
	private static void checkSyrk(MatrixTranspose transpose, int dimension, int shared,
			float[] matrix, float[] result) {
		if (transpose == null || dimension < 1 || shared < 1 || matrix == null
				|| matrix.length != dimension * shared || result == null
				|| result.length != dimension * dimension)
			throw new IllegalArgumentException("SYRK dimensions do not conform");
	}
	private static void checkTriangular(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		if (triangle == null || transpose == null || diagonal == null || dimension < 1
				|| matrix == null || matrix.length != dimension * dimension || vector == null
				|| vector.length != dimension) throw new IllegalArgumentException("TRSV dimensions do not conform");
	}
	private static void checkTriangular(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		if (triangle == null || transpose == null || diagonal == null || dimension < 1
				|| matrix == null || matrix.length != dimension * dimension || vector == null
				|| vector.length != dimension) throw new IllegalArgumentException("TRSV dimensions do not conform");
	}
	private static void checkTrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double[] matrix, double[] right) {
		int order = side == MatrixSide.LEFT ? rows : columns;
		if (side == null || triangle == null || transpose == null || diagonal == null
				|| rows < 1 || columns < 1 || matrix == null || matrix.length != order * order
				|| right == null || right.length != rows * columns)
			throw new IllegalArgumentException("TRSM dimensions do not conform");
	}
	private static void checkTrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float[] matrix, float[] right) {
		int order = side == MatrixSide.LEFT ? rows : columns;
		if (side == null || triangle == null || transpose == null || diagonal == null
				|| rows < 1 || columns < 1 || matrix == null || matrix.length != order * order
				|| right == null || right.length != rows * columns)
			throw new IllegalArgumentException("TRSM dimensions do not conform");
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
	private static void checkDecompositionMatrix(double[]matrix,int rows,int columns){if(rows<1||columns<1||matrix==null||matrix.length!=rows*columns)throw new IllegalArgumentException("invalid decomposition matrix dimensions");for(double value:matrix)if(!Double.isFinite(value))throw new IllegalArgumentException("decomposition matrix must be finite");}
	private static void checkDecompositionMatrix(float[]matrix,int rows,int columns){if(rows<1||columns<1||matrix==null||matrix.length!=rows*columns)throw new IllegalArgumentException("invalid FP32 decomposition matrix dimensions");for(float value:matrix)if(!Float.isFinite(value))throw new IllegalArgumentException("FP32 decomposition matrix must be finite");}
	private static void checkSymmetric(double[]matrix,int n){double scale=1;for(double value:matrix)scale=Math.max(scale,Math.abs(value));double tolerance=64*Math.ulp(1.0)*scale;for(int r=0;r<n;r++)for(int c=r+1;c<n;c++)if(Math.abs(matrix[r*n+c]-matrix[c*n+r])>tolerance)throw new IllegalArgumentException("eigenvalue matrix must be symmetric");}
	private static void checkSymmetric(float[]matrix,int n){float scale=1;for(float value:matrix)scale=Math.max(scale,Math.abs(value));float tolerance=64*Math.ulp(1.0f)*scale;for(int r=0;r<n;r++)for(int c=r+1;c<n;c++)if(Math.abs(matrix[r*n+c]-matrix[c*n+r])>tolerance)throw new IllegalArgumentException("FP32 eigenvalue matrix must be symmetric");}
	private static int[] shape(double[][] matrix) { if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) throw new IllegalArgumentException("matrix must be nonempty");
		int columns = matrix[0].length; for (double[] row : matrix) if (row == null || row.length != columns) throw new IllegalArgumentException("matrix must be rectangular"); return new int[] {matrix.length, columns}; }
	private static double[] flatten(double[][] matrix) { int[] shape = shape(matrix); double[] result = new double[shape[0] * shape[1]]; int offset = 0;
		for (double[] row : matrix) { System.arraycopy(row, 0, result, offset, row.length); offset += row.length; } return result; }
	private static double[][] reshape(double[] values, int rows, int columns) { double[][] result = new double[rows][columns];
		for (int row = 0; row < rows; row++) System.arraycopy(values, row * columns, result[row], 0, columns); return result; }
}
