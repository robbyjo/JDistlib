/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.vulkan;

import static org.lwjgl.util.shaderc.Shaderc.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memByteBuffer;
import static org.lwjgl.vulkan.VK10.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryHeap;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

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
import jdistlib.accelerator.PivotedQrFactor;
import jdistlib.accelerator.SingularValueDecomposition;
import jdistlib.accelerator.SymmetricEigenDecomposition;
import jdistlib.accelerator.UnaryOperation;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/**
 * Optional FP64 Vulkan 1.0 compute backend using LWJGL and runtime GLSL-to-SPIR-V
 * compilation. The provider deliberately uses host-visible storage so it can run
 * across discrete and integrated Vulkan devices without a vendor-specific copy path.
 */
public final class VulkanComputeBackend implements ComputeBackend {
	private static final int VECTOR_LOCAL_SIZE = 256;
	private static final String FLOAT_HEADER = "#version 450\n";
	private static final String HEADER = "#version 450\n"
			+ "#extension GL_ARB_gpu_shader_fp64 : require\n"
			+ "double pow2normal(int e){return packDouble2x32(uvec2(0u,uint(e+1023)<<20));}\n"
			+ "double pow2(int e){if(e>1023)return packDouble2x32(uvec2(0u,0x7ff00000u));"
			+ "if(e>=-1022)return pow2normal(e);if(e>=-1074)return pow2normal(-1022)*pow2normal(e+1022);return 0.0;}\n"
			+ "double dexp(double x){if(x>709.782712893384)return pow2(1024);if(x< -745.133219101941)return 0.0;"
			+ "double q=x*1.4426950408889634074;int k=int(q+(q>=0.0?0.5:-0.5));"
			+ "double r=(x-double(k)*0.69314718055994530942);double term=1.0,sum=1.0;"
			+ "for(int i=1;i<=20;i++){term*=r/double(i);sum+=term;}return sum*pow2(k);}\n"
			+ "double dlog(double x){if(x<=0.0)return x==0.0?-pow2(1024):packDouble2x32(uvec2(0u,0x7ff80000u));"
			+ "uvec2 bits=unpackDouble2x32(x);int e=int((bits.y>>20)&0x7ffu)-1023;"
			+ "if(e==-1023){x*=pow2normal(52);bits=unpackDouble2x32(x);e=int((bits.y>>20)&0x7ffu)-1023-52;}"
			+ "bits.y=(bits.y&0x000fffffu)|0x3ff00000u;double m=packDouble2x32(bits);"
			+ "double z=(m-1.0)/(m+1.0),z2=z*z,term=z,sum=z;for(int i=3;i<=41;i+=2){term*=z2;sum+=term/double(i);}"
			+ "return 2.0*sum+double(e)*0.69314718055994530942;}\n"
			+ "double dlog1p(double x){if(abs(x)>=0.0001)return dlog(1.0+x);double term=x,sum=x;"
			+ "for(int i=2;i<=24;i++){term*=-x;sum+=term/double(i);}return sum;}\n"
			+ "double dtanh(double x){if(x>20.0)return 1.0;if(x< -20.0)return -1.0;"
			+ "double e=dexp(2.0*abs(x)),v=(e-1.0)/(e+1.0);return x<0.0?-v:v;}\n"
			+ "double dlogistic(double x){return x>=0.0?1.0/(1.0+dexp(-x)):dexp(x)/(1.0+dexp(x));}\n";
	private static final String UNARY_SHADER = HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=1) writeonly buffer Y{double y[];};\n"
			+ "layout(push_constant) uniform P{int op;int n;}p;\n"
			+ "void main(){uint i=gl_GlobalInvocationID.x;if(i>=p.n)return;double v=x[i];"
			+ "if(p.op==0)y[i]=dexp(v);else if(p.op==1)y[i]=dlog(v);"
			+ "else if(p.op==2)y[i]=dlog1p(v);else if(p.op==3)y[i]=sqrt(v);"
			+ "else if(p.op==4)y[i]=dtanh(v);else y[i]=dlogistic(v);}\n";
	private static final String AXPY_SHADER = HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=1) readonly buffer Y{double y[];};\n"
			+ "layout(std430,binding=2) writeonly buffer Z{double z[];};\n"
			+ "layout(push_constant) uniform P{double alpha;int n;}p;\n"
			+ "void main(){uint i=gl_GlobalInvocationID.x;if(i<p.n)z[i]=p.alpha*x[i]+y[i];}\n";
	private static final String DOT_SHADER = HEADER
			+ "layout(local_size_x=1) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=1) readonly buffer Y{double y[];};\n"
			+ "layout(std430,binding=2) writeonly buffer O{double outValue[];};\n"
			+ "layout(push_constant) uniform P{int n;}p;\n"
			+ "void main(){double s=0.0;for(int i=0;i<p.n;i++)s+=x[i]*y[i];outValue[0]=s;}\n";
	private static final String GEMM_SHADER = HEADER
			+ "layout(local_size_x=16,local_size_y=16) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{double a[];};\n"
			+ "layout(std430,binding=1) readonly buffer B{double b[];};\n"
			+ "layout(std430,binding=2) writeonly buffer C{double c[];};\n"
			+ "layout(push_constant) uniform P{int m;int k;int n;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.y,col=gl_GlobalInvocationID.x;"
			+ "if(row>=p.m||col>=p.n)return;double s=0.0;for(int q=0;q<p.k;q++)s+=a[row*p.k+q]*b[q*p.n+col];c[row*p.n+col]=s;}\n";
	private static final String BLAS_NRM2_SHADER = HEADER
			+ "layout(local_size_x=1) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=1) writeonly buffer O{double outValue[];};\n"
			+ "layout(push_constant) uniform P{int n;int offset;int stride;}p;\n"
			+ "void main(){double scale=0.0,s=1.0;for(int i=0;i<p.n;i++){double v=abs(x[p.offset+i*p.stride]);"
			+ "if(v!=0.0){if(scale<v){double r=scale/v;s=1.0+s*r*r;scale=v;}else{double r=v/scale;s+=r*r;}}}"
			+ "outValue[0]=scale==0.0?0.0:scale*sqrt(s);}\n";
	private static final String BLAS_GEMV_SHADER = HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{double a[];};\n"
			+ "layout(std430,binding=1) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=2) buffer Y{double y[];};\n"
			+ "layout(push_constant) uniform P{double alpha;double beta;int tr;int rows;int cols;}p;\n"
			+ "void main(){uint i=gl_GlobalInvocationID.x;int outCount=p.tr!=0?p.cols:p.rows,inCount=p.tr!=0?p.rows:p.cols;"
			+ "if(i>=outCount)return;double s=0.0;for(int j=0;j<inCount;j++)s+=(p.tr!=0?a[j*p.cols+i]:a[i*p.cols+j])*x[j];"
			+ "y[i]=p.alpha*s+p.beta*y[i];}\n";
	private static final String BLAS_GEMM_SHADER = HEADER
			+ "layout(local_size_x=16,local_size_y=16) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{double a[];};\n"
			+ "layout(std430,binding=1) readonly buffer B{double b[];};\n"
			+ "layout(std430,binding=2) buffer C{double c[];};\n"
			+ "layout(push_constant) uniform P{double alpha;double beta;int ta;int tb;int m;int n;int k;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.y,col=gl_GlobalInvocationID.x;if(row>=p.m||col>=p.n)return;double s=0.0;"
			+ "for(int q=0;q<p.k;q++)s+=(p.ta!=0?a[q*p.m+row]:a[row*p.k+q])*(p.tb!=0?b[col*p.k+q]:b[q*p.n+col]);"
			+ "int z=int(row)*p.n+int(col);c[z]=p.alpha*s+p.beta*c[z];}\n";
	private static final String CSR_MV_SHADER = HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer V{double v[];};\n"
			+ "layout(std430,binding=1) readonly buffer CI{double ci[];};\n"
			+ "layout(std430,binding=2) readonly buffer RS{double rs[];};\n"
			+ "layout(std430,binding=3) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=4) buffer Y{double y[];};\n"
			+ "layout(push_constant) uniform P{double alpha;double beta;int rows;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.x;if(row>=p.rows)return;double s=0.0;"
			+ "for(int z=int(rs[row])-1;z<int(rs[row+1])-1;z++)s+=v[z]*x[int(ci[z])-1];y[row]=p.alpha*s+p.beta*y[row];}\n";
	private static final String CSR_MM_SHADER = HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer V{double v[];};\n"
			+ "layout(std430,binding=1) readonly buffer CI{double ci[];};\n"
			+ "layout(std430,binding=2) readonly buffer RS{double rs[];};\n"
			+ "layout(std430,binding=3) readonly buffer B{double b[];};\n"
			+ "layout(std430,binding=4) buffer C{double c[];};\n"
			+ "layout(push_constant) uniform P{double alpha;double beta;int rows;int cols;}p;\n"
			+ "void main(){uint z=gl_GlobalInvocationID.x;if(z>=p.rows*p.cols)return;int row=int(z)/p.cols,col=int(z)-row*p.cols;double s=0.0;"
			+ "for(int q=int(rs[row])-1;q<int(rs[row+1])-1;q++)s+=v[q]*b[(int(ci[q])-1)*p.cols+col];c[z]=p.alpha*s+p.beta*c[z];}\n";
	private static final String FLOAT_AXPY_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{float x[];};\n"
			+ "layout(std430,binding=1) buffer Y{float y[];};\n"
			+ "layout(push_constant) uniform P{int n;float alpha;int xo;int xs;int yo;int ys;}p;\n"
			+ "void main(){uint i=gl_GlobalInvocationID.x;if(i<p.n)y[p.yo+int(i)*p.ys]=p.alpha*x[p.xo+int(i)*p.xs]+y[p.yo+int(i)*p.ys];}\n";
	private static final String FLOAT_DOT_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=1) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{float x[];};\n"
			+ "layout(std430,binding=1) readonly buffer Y{float y[];};\n"
			+ "layout(std430,binding=2) writeonly buffer O{float outValue[];};\n"
			+ "layout(push_constant) uniform P{int n;int xo;int xs;int yo;int ys;}p;\n"
			+ "void main(){float s=0.0;for(int i=0;i<p.n;i++)s+=x[p.xo+i*p.xs]*y[p.yo+i*p.ys];outValue[0]=s;}\n";
	private static final String FLOAT_NRM2_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=1) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{float x[];};\n"
			+ "layout(std430,binding=1) writeonly buffer O{float outValue[];};\n"
			+ "layout(push_constant) uniform P{int n;int offset;int stride;}p;\n"
			+ "void main(){float scale=0.0,s=1.0;for(int i=0;i<p.n;i++){float v=abs(x[p.offset+i*p.stride]);"
			+ "if(v!=0.0){if(scale<v){float r=scale/v;s=1.0+s*r*r;scale=v;}else{float r=v/scale;s+=r*r;}}}"
			+ "outValue[0]=scale==0.0?0.0:scale*sqrt(s);}\n";
	private static final String FLOAT_GEMV_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{float a[];};\n"
			+ "layout(std430,binding=1) readonly buffer X{float x[];};\n"
			+ "layout(std430,binding=2) buffer Y{float y[];};\n"
			+ "layout(push_constant) uniform P{float alpha;float beta;int tr;int rows;int cols;}p;\n"
			+ "void main(){uint i=gl_GlobalInvocationID.x;int outCount=p.tr!=0?p.cols:p.rows,inCount=p.tr!=0?p.rows:p.cols;"
			+ "if(i>=outCount)return;float s=0.0;for(int j=0;j<inCount;j++)s+=(p.tr!=0?a[j*p.cols+i]:a[i*p.cols+j])*x[j];"
			+ "y[i]=p.alpha*s+p.beta*y[i];}\n";
	private static final String FLOAT_GEMM_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=16,local_size_y=16) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{float a[];};\n"
			+ "layout(std430,binding=1) readonly buffer B{float b[];};\n"
			+ "layout(std430,binding=2) buffer C{float c[];};\n"
			+ "layout(push_constant) uniform P{float alpha;float beta;int ta;int tb;int m;int n;int k;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.y,col=gl_GlobalInvocationID.x;if(row>=p.m||col>=p.n)return;float s=0.0;"
			+ "for(int q=0;q<p.k;q++)s+=(p.ta!=0?a[q*p.m+row]:a[row*p.k+q])*(p.tb!=0?b[col*p.k+q]:b[q*p.n+col]);"
			+ "int z=int(row)*p.n+int(col);c[z]=p.alpha*s+p.beta*c[z];}\n";
	private static final String FLOAT_CSR_MV_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer V{float v[];};\n"
			+ "layout(std430,binding=1) readonly buffer CI{int ci[];};\n"
			+ "layout(std430,binding=2) readonly buffer RS{int rs[];};\n"
			+ "layout(std430,binding=3) readonly buffer X{float x[];};\n"
			+ "layout(std430,binding=4) buffer Y{float y[];};\n"
			+ "layout(push_constant) uniform P{float alpha;float beta;int rows;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.x;if(row>=p.rows)return;float s=0.0;"
			+ "for(int z=rs[row]-1;z<rs[row+1]-1;z++)s+=v[z]*x[ci[z]-1];y[row]=p.alpha*s+p.beta*y[row];}\n";
	private static final String FLOAT_CSR_MM_SHADER = FLOAT_HEADER
			+ "layout(local_size_x=256) in;\n"
			+ "layout(std430,binding=0) readonly buffer V{float v[];};\n"
			+ "layout(std430,binding=1) readonly buffer CI{int ci[];};\n"
			+ "layout(std430,binding=2) readonly buffer RS{int rs[];};\n"
			+ "layout(std430,binding=3) readonly buffer B{float b[];};\n"
			+ "layout(std430,binding=4) buffer C{float c[];};\n"
			+ "layout(push_constant) uniform P{float alpha;float beta;int rows;int cols;}p;\n"
			+ "void main(){uint z=gl_GlobalInvocationID.x;if(z>=p.rows*p.cols)return;int row=int(z)/p.cols,col=int(z)-row*p.cols;float s=0.0;"
			+ "for(int q=rs[row]-1;q<rs[row+1]-1;q++)s+=v[q]*b[(ci[q]-1)*p.cols+col];c[z]=p.alpha*s+p.beta*c[z];}\n";
	private static final String BLAS_SYRK_SHADER = syrkShader(HEADER, "double");
	private static final String BLAS_TRSV_SHADER = trsvShader(HEADER, "double");
	private static final String BLAS_TRSM_SHADER = trsmShader(HEADER, "double");
	private static final String FLOAT_SYRK_SHADER = syrkShader(FLOAT_HEADER, "float");
	private static final String FLOAT_TRSV_SHADER = trsvShader(FLOAT_HEADER, "float");
	private static final String FLOAT_TRSM_SHADER = trsmShader(FLOAT_HEADER, "float");
	private static final String DPOTRF_SHADER = choleskyShader(HEADER,"double");
	private static final String SPOTRF_SHADER = choleskyShader(FLOAT_HEADER,"float");
	private static final String DGEQP3_SHADER = qrShader(HEADER,"double");
	private static final String SGEQP3_SHADER = qrShader(FLOAT_HEADER,"float");
	private static final String DSYEV_SHADER = eigenShader(HEADER,"double");
	private static final String SSYEV_SHADER = eigenShader(FLOAT_HEADER,"float");
	private static final String DGESVD_SHADER = svdShader(HEADER,"double");
	private static final String SGESVD_SHADER = svdShader(FLOAT_HEADER,"float");
	private static final String LOGISTIC_SHADER = HEADER
			+ "layout(local_size_x=64) in;\n"
			+ "layout(std430,binding=0) readonly buffer X{double x[];};\n"
			+ "layout(std430,binding=1) readonly buffer Y{double y[];};\n"
			+ "layout(std430,binding=2) readonly buffer Q{double q[];};\n"
			+ "layout(std430,binding=3) writeonly buffer V{double value[];};\n"
			+ "layout(std430,binding=4) writeonly buffer G{double gradient[];};\n"
			+ "layout(push_constant) uniform P{double prior;int rows;int dims;int chains;}p;\n"
			+ "double logistic(double z){return dlogistic(z);}\n"
			+ "double l1e(double z){return z>0.0?z+dlog1p(dexp(-z)):dlog1p(dexp(z));}\n"
			+ "void main(){uint chain=gl_GlobalInvocationID.x;if(chain>=p.chains)return;double lp=0.0;"
			+ "for(int d=0;d<p.dims;d++){double state=q[chain*p.dims+d];lp-=0.5*p.prior*state*state;gradient[chain*p.dims+d]=-p.prior*state;}"
			+ "for(int row=0;row<p.rows;row++){double eta=0.0;for(int d=0;d<p.dims;d++)eta+=x[row*p.dims+d]*q[chain*p.dims+d];"
			+ "lp+=y[row]*eta-l1e(eta);double residual=y[row]-logistic(eta);for(int d=0;d<p.dims;d++)gradient[chain*p.dims+d]+=residual*x[row*p.dims+d];}value[chain]=lp;}\n";

	private static String syrkShader(String header, String type) { return header
			+ "layout(local_size_x=16,local_size_y=16) in;\n"
			+ "layout(std430,binding=0) readonly buffer A{" + type + " a[];};\n"
			+ "layout(std430,binding=1) buffer C{" + type + " c[];};\n"
			+ "layout(push_constant) uniform P{" + type + " alpha;" + type + " beta;int tr;int n;int k;}p;\n"
			+ "void main(){uint row=gl_GlobalInvocationID.y,col=gl_GlobalInvocationID.x;if(row>=p.n||col>=p.n)return;"
			+ type + " s=0;for(int q=0;q<p.k;q++)s+=(p.tr!=0?a[q*p.n+row]:a[row*p.k+q])*(p.tr!=0?a[q*p.n+col]:a[col*p.k+q]);"
			+ "c[row*p.n+col]=p.alpha*s+p.beta*c[row*p.n+col];}\n"; }
	private static String trsvShader(String header, String type) { return header
			+ "layout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer A{" + type + " a[];};\n"
			+ "layout(std430,binding=1) buffer X{" + type + " x[];};\n"
			+ "layout(push_constant) uniform P{int lower;int tr;int unit;int n;}p;\n"
			+ "void main(){bool effective=p.tr!=0?p.lower==0:p.lower!=0;for(int step=0;step<p.n;step++){int i=effective?step:p.n-1-step;"
			+ type + " v=x[i];if(effective){for(int j=0;j<i;j++)v-=(p.tr!=0?a[j*p.n+i]:a[i*p.n+j])*x[j];}"
			+ "else{for(int j=i+1;j<p.n;j++)v-=(p.tr!=0?a[j*p.n+i]:a[i*p.n+j])*x[j];}x[i]=p.unit!=0?v:v/a[i*p.n+i];}}\n"; }
	private static String trsmShader(String header, String type) { return header
			+ "layout(local_size_x=256) in;\nlayout(std430,binding=0) readonly buffer A{" + type + " a[];};\n"
			+ "layout(std430,binding=1) buffer B{" + type + " b[];};\n"
			+ "layout(push_constant) uniform P{" + type + " alpha;int side;int lower;int tr;int unit;int rows;int cols;}p;\n"
			+ "void main(){int vector=int(gl_GlobalInvocationID.x),count=p.side!=0?p.rows:p.cols,n=p.side!=0?p.cols:p.rows;if(vector>=count)return;"
			+ "if(p.side==0){bool effective=p.tr!=0?p.lower==0:p.lower!=0;for(int i=0;i<n;i++)b[i*p.cols+vector]*=p.alpha;"
			+ "for(int step=0;step<n;step++){int i=effective?step:n-1-step;" + type + " v=b[i*p.cols+vector];"
			+ "if(effective){for(int j=0;j<i;j++)v-=(p.tr!=0?a[j*n+i]:a[i*n+j])*b[j*p.cols+vector];}"
			+ "else{for(int j=i+1;j<n;j++)v-=(p.tr!=0?a[j*n+i]:a[i*n+j])*b[j*p.cols+vector];}b[i*p.cols+vector]=p.unit!=0?v:v/a[i*n+i];}}"
			+ "else{bool effective=p.tr!=0?p.lower==0:p.lower!=0;for(int j=0;j<n;j++)b[vector*p.cols+j]*=p.alpha;"
			+ "for(int step=0;step<n;step++){int j=effective?n-1-step:step;" + type + " v=b[vector*p.cols+j];"
			+ "if(effective){for(int k=j+1;k<n;k++)v-=b[vector*p.cols+k]*(p.tr!=0?a[j*n+k]:a[k*n+j]);}"
			+ "else{for(int k=0;k<j;k++)v-=b[vector*p.cols+k]*(p.tr!=0?a[j*n+k]:a[k*n+j]);}b[vector*p.cols+j]=p.unit!=0?v:v/a[j*n+j];}}}\n"; }

	private static String choleskyShader(String header,String type){return header
			+"layout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer A{"+type+" a[];};\n"
			+"layout(std430,binding=1) buffer L{"+type+" l[];};\nlayout(std430,binding=2) buffer I{int info[];};\n"
			+"layout(push_constant) uniform P{int n;}p;\nvoid main(){info[0]=0;for(int i=0;i<p.n*p.n;i++)l[i]=0;for(int r=0;r<p.n;r++)for(int c=0;c<=r;c++){"+type+" s=a[r*p.n+c];for(int k=0;k<c;k++)s-=l[r*p.n+k]*l[c*p.n+k];if(r==c){if(!(s>0)){info[0]=r+1;return;}l[r*p.n+c]=sqrt(s);}else l[r*p.n+c]=s/l[c*p.n+c];}}\n";}
	private static String qrShader(String header,String type){return header
			+"layout(local_size_x=1) in;\nlayout(std430,binding=0) buffer Q{"+type+" qr[];};\nlayout(std430,binding=1) buffer T{"+type+" tau[];};\nlayout(std430,binding=2) buffer Piv{int pivot[];};\nlayout(push_constant) uniform P{int rows;int cols;}p;\n"
			+type+" absv("+type+" x){return x<0?-x:x;}\nvoid swapcols(int first,int second){for(int r=0;r<p.rows;r++){int a=r*p.cols+first,b=r*p.cols+second;"+type+" z=qr[a];qr[a]=qr[b];qr[b]=z;}}\n"
			+"void main(){int count=min(p.rows,p.cols);for(int i=0;i<p.cols;i++)pivot[i]=i;for(int k=0;k<count;k++){int selected=k;"+type+" best=-1;for(int c=k;c<p.cols;c++){"+type+" norm=0;for(int r=k;r<p.rows;r++){"+type+" v=qr[r*p.cols+c];norm+=v*v;}if(norm>best){best=norm;selected=c;}}if(selected!=k){swapcols(k,selected);int z=pivot[k];pivot[k]=pivot[selected];pivot[selected]=z;}"+type+" scale=0,sum=1;for(int r=k;r<p.rows;r++){"+type+" v=absv(qr[r*p.cols+k]);if(v!=0){if(scale<v){"+type+" z=scale/v;sum=1+sum*z*z;scale=v;}else{"+type+" z=v/scale;sum+=z*z;}}}"+type+" norm=scale==0?0:scale*sqrt(sum);if(norm==0){tau[k]=0;continue;}"+type+" alpha=qr[k*p.cols+k],diagonal=alpha<0?norm:-norm;tau[k]=(diagonal-alpha)/diagonal;"+type+" denominator=alpha-diagonal;qr[k*p.cols+k]=diagonal;for(int r=k+1;r<p.rows;r++)qr[r*p.cols+k]/=denominator;for(int c=k+1;c<p.cols;c++){"+type+" product=qr[k*p.cols+c];for(int r=k+1;r<p.rows;r++)product+=qr[r*p.cols+k]*qr[r*p.cols+c];product*=tau[k];qr[k*p.cols+c]-=product;for(int r=k+1;r<p.rows;r++)qr[r*p.cols+c]-=qr[r*p.cols+k]*product;}}}\n";}
	private static String eigenShader(String header,String type){return header
			+"layout(local_size_x=1) in;\nlayout(std430,binding=0) buffer A{"+type+" a[];};\nlayout(std430,binding=1) buffer E{"+type+" eigenvalues[];};\nlayout(std430,binding=2) buffer V{"+type+" vectors[];};\nlayout(std430,binding=3) buffer I{int info[];};\n"
			+"layout(push_constant) uniform P{"+type+" eps;int n;}p;\n"+type+" absv("+type+" x){return x<0?-x:x;}\nvoid swapcols(int first,int second){for(int r=0;r<p.n;r++){int x=r*p.n+first,y=r*p.n+second;"+type+" z=vectors[x];vectors[x]=vectors[y];vectors[y]=z;}}\n"
			+"void main(){info[0]=0;for(int i=0;i<p.n*p.n;i++)vectors[i]=0;for(int i=0;i<p.n;i++)vectors[i*p.n+i]=1;int maximum=max(32,8*p.n);for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int x=0;x<p.n-1;x++)for(int y=x+1;y<p.n;y++){"+type+" apq=a[x*p.n+y],threshold=p.eps*(absv(a[x*p.n+x])+absv(a[y*p.n+y])+1);if(absv(apq)<=threshold)continue;changed=true;"+type+" app=a[x*p.n+x],aqq=a[y*p.n+y],tau=(aqq-app)/(2*apq),t=(tau<0?-1:1)/(absv(tau)+sqrt(1+tau*tau)),c=1/sqrt(1+t*t),s=t*c;for(int k=0;k<p.n;k++)if(k!=x&&k!=y){"+type+" akp=a[k*p.n+x],akq=a[k*p.n+y],np=c*akp-s*akq,nq=s*akp+c*akq;a[k*p.n+x]=a[x*p.n+k]=np;a[k*p.n+y]=a[y*p.n+k]=nq;}a[x*p.n+x]=c*c*app-2*s*c*apq+s*s*aqq;a[y*p.n+y]=s*s*app+2*s*c*apq+c*c*aqq;a[x*p.n+y]=a[y*p.n+x]=0;for(int r=0;r<p.n;r++){"+type+" vp=vectors[r*p.n+x],vq=vectors[r*p.n+y];vectors[r*p.n+x]=c*vp-s*vq;vectors[r*p.n+y]=s*vp+c*vq;}}if(!changed)break;if(sweep+1==maximum){info[0]=1;return;}}for(int i=0;i<p.n;i++)eigenvalues[i]=a[i*p.n+i];for(int i=0;i<p.n-1;i++){int selected=i;for(int j=i+1;j<p.n;j++)if(eigenvalues[j]<eigenvalues[selected])selected=j;if(selected!=i){"+type+" z=eigenvalues[i];eigenvalues[i]=eigenvalues[selected];eigenvalues[selected]=z;swapcols(i,selected);}}for(int c=0;c<p.n;c++){int largest=0;for(int r=1;r<p.n;r++)if(absv(vectors[r*p.n+c])>absv(vectors[largest*p.n+c]))largest=r;if(vectors[largest*p.n+c]<0)for(int r=0;r<p.n;r++)vectors[r*p.n+c]=-vectors[r*p.n+c];}}\n";}
	private static String svdShader(String header,String type){return header
			+"layout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer A{"+type+" a[];};\nlayout(std430,binding=1) buffer W{"+type+" work[];};\nlayout(std430,binding=2) buffer U{"+type+" u[];};\nlayout(std430,binding=3) buffer S{"+type+" singular[];};\nlayout(std430,binding=4) buffer VT{"+type+" vt[];};\nlayout(std430,binding=5) buffer I{int info[];};\nlayout(push_constant) uniform P{"+type+" eps;int rows;int cols;}p;\n"
			+type+" absv("+type+" x){return x<0?-x:x;}\n"+type+" normWork(int rows,int cols,int col){"+type+" scale=0,sum=1;for(int r=0;r<rows;r++){"+type+" v=absv(work[r*cols+col]);if(v!=0){if(scale<v){"+type+" z=scale/v;sum=1+sum*z*z;scale=v;}else{"+type+" z=v/scale;sum+=z*z;}}}return scale==0?0:scale*sqrt(sum);}\n"
			+"void swapWork(int rows,int cols,int x,int y){for(int r=0;r<rows;r++){int i=r*cols+x,j=r*cols+y;"+type+" z=work[i];work[i]=work[j];work[j]=z;}}\nvoid swapV(bool wide,int n,int x,int y){for(int r=0;r<n;r++){int i=r*n+x,j=r*n+y;"+type+" z=wide?u[i]:vt[i];if(wide){u[i]=u[j];u[j]=z;}else{vt[i]=vt[j];vt[j]=z;}}}\n"
			+"bool completeU(int rows,int cols,int col){for(int candidate=0;candidate<rows;candidate++){for(int r=0;r<rows;r++)u[r*cols+col]=r==candidate?1:0;for(int q=0;q<col;q++){"+type+" product=0;for(int r=0;r<rows;r++)product+=u[r*cols+q]*u[r*cols+col];for(int r=0;r<rows;r++)u[r*cols+col]-=product*u[r*cols+q];}"+type+" scale=0,sum=1;for(int r=0;r<rows;r++){"+type+" v=absv(u[r*cols+col]);if(v!=0){if(scale<v){"+type+" z=scale/v;sum=1+sum*z*z;scale=v;}else{"+type+" z=v/scale;sum+=z*z;}}}" + type + " norm=scale==0?0:scale*sqrt(sum);if(norm>p.eps){for(int r=0;r<rows;r++)u[r*cols+col]/=norm;return true;}}return false;}\n"
			+"bool completeWork(int rows,int cols,int col){for(int candidate=0;candidate<rows;candidate++){for(int r=0;r<rows;r++)work[r*cols+col]=r==candidate?1:0;for(int q=0;q<col;q++){"+type+" product=0;for(int r=0;r<rows;r++)product+=work[r*cols+q]*work[r*cols+col];for(int r=0;r<rows;r++)work[r*cols+col]-=product*work[r*cols+q];}"+type+" norm=normWork(rows,cols,col);if(norm>p.eps){for(int r=0;r<rows;r++)work[r*cols+col]/=norm;return true;}}return false;}\n"
			+"void main(){info[0]=0;bool wide=p.rows<p.cols;int tr=wide?p.cols:p.rows,tc=wide?p.rows:p.cols;for(int r=0;r<tr;r++)for(int c=0;c<tc;c++)work[r*tc+c]=wide?a[c*p.cols+r]:a[r*p.cols+c];for(int i=0;i<tc*tc;i++){if(wide)u[i]=0;else vt[i]=0;}for(int i=0;i<tc;i++){if(wide)u[i*tc+i]=1;else vt[i*tc+i]=1;}int maximum=max(48,12*tc);for(int sweep=0;sweep<maximum;sweep++){bool changed=false;for(int x=0;x<tc-1;x++)for(int y=x+1;y<tc;y++){"+type+" alpha=0,beta=0,gamma=0;for(int r=0;r<tr;r++){"+type+" left=work[r*tc+x],right=work[r*tc+y];alpha+=left*left;beta+=right*right;gamma+=left*right;}if(gamma==0||absv(gamma)<=p.eps*sqrt(alpha*beta))continue;changed=true;"+type+" zeta=(beta-alpha)/(2*gamma),t=(zeta<0?-1:1)/(absv(zeta)+sqrt(1+zeta*zeta)),c=1/sqrt(1+t*t),s=c*t;for(int r=0;r<tr;r++){int i=r*tc+x,j=r*tc+y;"+type+" left=work[i],right=work[j];work[i]=c*left-s*right;work[j]=s*left+c*right;}for(int r=0;r<tc;r++){int i=r*tc+x,j=r*tc+y;"+type+" left=wide?u[i]:vt[i],right=wide?u[j]:vt[j];if(wide){u[i]=c*left-s*right;u[j]=s*left+c*right;}else{vt[i]=c*left-s*right;vt[j]=s*left+c*right;}}}if(!changed)break;if(sweep+1==maximum){info[0]=1;return;}}for(int c=0;c<tc;c++)singular[c]=normWork(tr,tc,c);for(int i=0;i<tc-1;i++){int selected=i;for(int j=i+1;j<tc;j++)if(singular[j]>singular[selected])selected=j;if(selected!=i){"+type+" z=singular[i];singular[i]=singular[selected];singular[selected]=z;swapWork(tr,tc,i,selected);swapV(wide,tc,i,selected);}}"+type+" threshold=max(p.rows,p.cols)*p.eps/16*singular[0];if(!wide){for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)u[r*tc+c]=work[r*tc+c]/singular[c];else if(!completeU(tr,tc,c)){info[0]=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tr;r++)if(absv(u[r*tc+c])>absv(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tr;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tc;r++)vt[r*tc+c]=-vt[r*tc+c];}}for(int r=0;r<tc;r++)for(int c=r+1;c<tc;c++){"+type+" z=vt[r*tc+c];vt[r*tc+c]=vt[c*tc+r];vt[c*tc+r]=z;}}else{for(int c=0;c<tc;c++){if(singular[c]>threshold)for(int r=0;r<tr;r++)work[r*tc+c]/=singular[c];else if(!completeWork(tr,tc,c)){info[0]=2;return;}}for(int c=0;c<tc;c++){int largest=0;for(int r=1;r<tc;r++)if(absv(u[r*tc+c])>absv(u[largest*tc+c]))largest=r;if(u[largest*tc+c]<0){for(int r=0;r<tc;r++)u[r*tc+c]=-u[r*tc+c];for(int r=0;r<tr;r++)work[r*tc+c]=-work[r*tc+c];}}for(int c=0;c<tc;c++)for(int r=0;r<tr;r++)vt[c*tr+r]=work[r*tc+c];}}\n";}

	private VkInstance instance;
	private VkPhysicalDevice physicalDevice;
	private VkDevice device;
	private VkQueue queue;
	private int queueFamily = -1;
	private long commandPool;
	private Kernel unaryKernel, axpyKernel, dotKernel, gemmKernel, logisticKernel;
	private Kernel blasNrm2Kernel, blasGemvKernel, blasGemmKernel, blasSyrkKernel;
	private Kernel blasTrsvKernel, blasTrsmKernel, csrMvKernel, csrMmKernel;
	private Kernel floatAxpyKernel, floatDotKernel, floatNrm2Kernel, floatGemvKernel;
	private Kernel floatGemmKernel, floatSyrkKernel, floatTrsvKernel, floatTrsmKernel;
	private Kernel floatCsrMvKernel, floatCsrMmKernel;
	private Kernel dpotrfKernel, dgeqp3Kernel, dsyevKernel, dgesvdKernel;
	private Kernel spotrfKernel, sgeqp3Kernel, ssyevKernel, sgesvdKernel;
	private ComputeCapabilities capabilities;
	private ComputeDeviceInfo deviceInfo;
	private Throwable unavailableCause;

	/** Detects the first compute-capable FP64 Vulkan device. */
	public VulkanComputeBackend() {
		try {
			// Some drivers expose enough extensions to overflow LWJGL's 64 KiB default
			// while it constructs instance capabilities.
			Configuration.STACK_SIZE.set(1024);
			initialize();
		} catch (Throwable error) { unavailableCause = error; close(); }
	}

	@Override public String id() { return "vulkan"; }
	@Override public boolean available() { return unavailableCause == null && device != null; }
	@Override public ComputeCapabilities capabilities() { ensureAvailable(); return capabilities; }
	@Override public ComputeDeviceInfo deviceInfo() { ensureAvailable(); return deviceInfo; }
	/** @return the initialization failure, or {@code null} when Vulkan is available */
	public Throwable unavailableCause() { return unavailableCause; }

	@Override public synchronized double[] unary(UnaryOperation operation, double[] input) {
		if (operation == null || input == null) throw new IllegalArgumentException("operation and input required");
		ensureAvailable(); BufferResource x = create(input), y = createDoubles(input.length);
		try {
			ByteBuffer push = nativeBuffer(8).putInt(0, operation.ordinal()).putInt(4, input.length);
			if (unaryKernel == null) unaryKernel = new Kernel(UNARY_SHADER, 2, 8);
			execute(unaryKernel, resources(x, y), push, groups(input.length, VECTOR_LOCAL_SIZE), 1, 1);
			return y.readDoubles(input.length);
		} finally { y.close(); x.close(); }
	}

	@Override public synchronized double[] axpy(double alpha, double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); BufferResource bx = create(x), by = create(y), out = createDoubles(x.length);
		try {
			ByteBuffer push = nativeBuffer(16).putDouble(0, alpha).putInt(8, x.length);
			if (axpyKernel == null) axpyKernel = new Kernel(AXPY_SHADER, 3, 16);
			execute(axpyKernel, resources(bx, by, out), push, groups(x.length, VECTOR_LOCAL_SIZE), 1, 1);
			return out.readDoubles(x.length);
		} finally { out.close(); by.close(); bx.close(); }
	}

	@Override public synchronized double dot(double[] x, double[] y) {
		checkVectors(x, y); ensureAvailable(); BufferResource bx = create(x), by = create(y), out = createDoubles(1);
		try {
			ByteBuffer push = nativeBuffer(4).putInt(0, x.length);
			if (dotKernel == null) dotKernel = new Kernel(DOT_SHADER, 3, 4);
			execute(dotKernel, resources(bx, by, out), push, 1, 1, 1);
			return out.readDoubles(1)[0];
		} finally { out.close(); by.close(); bx.close(); }
	}

	@Override public synchronized double[][] matrixMultiply(double[][] left, double[][] right) {
		int[] aShape = shape(left), bShape = shape(right);
		if (aShape[1] != bShape[0]) throw new IllegalArgumentException("matrix dimensions do not conform");
		ensureAvailable(); BufferResource a = create(flatten(left)), b = create(flatten(right));
		BufferResource out = createDoubles(aShape[0] * bShape[1]);
		try {
			ByteBuffer push = nativeBuffer(12).putInt(0, aShape[0]).putInt(4, aShape[1]).putInt(8, bShape[1]);
			if (gemmKernel == null) gemmKernel = new Kernel(GEMM_SHADER, 3, 12);
			execute(gemmKernel, resources(a, b, out), push, groups(bShape[1], 16), groups(aShape[0], 16), 1);
			return reshape(out.readDoubles(aShape[0] * bShape[1]), aShape[0], bShape[1]);
		} finally { out.close(); b.close(); a.close(); }
	}

	@Override public synchronized void daxpy(int count, double alpha, double[] x,
			int xOffset, int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return; double[] packedX = new double[count], packedY = new double[count];
		for (int i = 0; i < count; i++) { packedX[i] = x[xOffset + i * xStride]; packedY[i] = y[yOffset + i * yStride]; }
		double[] updated = axpy(alpha, packedX, packedY);
		for (int i = 0; i < count; i++) y[yOffset + i * yStride] = updated[i];
	}

	@Override public synchronized double ddot(int count, double[] x, int xOffset,
			int xStride, double[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0; double[] packedX = new double[count], packedY = new double[count];
		for (int i = 0; i < count; i++) { packedX[i] = x[xOffset + i * xStride]; packedY[i] = y[yOffset + i * yStride]; }
		return dot(packedX, packedY);
	}

	@Override public synchronized double dnrm2(int count, double[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0; ensureAvailable();
		BufferResource input = create(x), output = createDoubles(1);
		try {
			ByteBuffer push = nativeBuffer(12).putInt(0, count).putInt(4, offset).putInt(8, stride);
			if (blasNrm2Kernel == null) blasNrm2Kernel = new Kernel(BLAS_NRM2_SHADER, 2, 12);
			execute(blasNrm2Kernel, resources(input, output), push, 1, 1, 1); return output.readDoubles(1)[0];
		} finally { output.close(); input.close(); }
	}

	@Override public synchronized void dgemv(MatrixTranspose transpose, int rows, int columns,
			double alpha, double[] matrix, double[] x, double beta, double[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable();
		BufferResource a = create(matrix), bx = create(x), by = create(y);
		try {
			ByteBuffer push = nativeBuffer(32).putDouble(0, alpha).putDouble(8, beta)
					.putInt(16, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0).putInt(20, rows).putInt(24, columns);
			if (blasGemvKernel == null) blasGemvKernel = new Kernel(BLAS_GEMV_SHADER, 3, 32);
			execute(blasGemvKernel, resources(a, bx, by), push, groups(y.length, VECTOR_LOCAL_SIZE), 1, 1);
			double[] updated = by.readDoubles(y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { by.close(); bx.close(); a.close(); }
	}

	@Override public synchronized void dgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, double alpha, double[] left, double[] right,
			double beta, double[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result); ensureAvailable();
		BufferResource a = create(left), b = create(right), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(48).putDouble(0, alpha).putDouble(8, beta)
					.putInt(16, leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(20, rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(24, rows).putInt(28, columns).putInt(32, shared);
			if (blasGemmKernel == null) blasGemmKernel = new Kernel(BLAS_GEMM_SHADER, 3, 48);
			execute(blasGemmKernel, resources(a, b, c), push, groups(columns, 16), groups(rows, 16), 1);
			double[] updated = c.readDoubles(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); b.close(); a.close(); }
	}
	@Override public synchronized void dsyrk(MatrixTranspose transpose, int dimension, int shared,
			double alpha, double[] matrix, double beta, double[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result); ensureAvailable();
		BufferResource a = create(matrix), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(32).putDouble(0, alpha).putDouble(8, beta)
					.putInt(16, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(20, dimension).putInt(24, shared);
			if (blasSyrkKernel == null) blasSyrkKernel = new Kernel(BLAS_SYRK_SHADER, 2, 32);
			execute(blasSyrkKernel, resources(a, c), push, groups(dimension, 16), groups(dimension, 16), 1);
			double[] updated = c.readDoubles(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); a.close(); }
	}
	@Override public synchronized void dtrsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, double[] matrix, double[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector); ensureAvailable();
		BufferResource a = create(matrix), x = create(vector);
		try {
			ByteBuffer push = nativeBuffer(16).putInt(0, triangle == MatrixTriangle.LOWER ? 1 : 0)
					.putInt(4, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(8, diagonal == MatrixDiagonal.UNIT ? 1 : 0).putInt(12, dimension);
			if (blasTrsvKernel == null) blasTrsvKernel = new Kernel(BLAS_TRSV_SHADER, 2, 16);
			execute(blasTrsvKernel, resources(a, x), push, 1, 1, 1);
			double[] updated = x.readDoubles(vector.length); System.arraycopy(updated, 0, vector, 0, vector.length);
		} finally { x.close(); a.close(); }
	}
	@Override public synchronized void dtrsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			double alpha, double[] matrix, double[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right); ensureAvailable();
		BufferResource a = create(matrix), b = create(right);
		try {
			ByteBuffer push = nativeBuffer(32).putDouble(0, alpha)
					.putInt(8, side == MatrixSide.RIGHT ? 1 : 0)
					.putInt(12, triangle == MatrixTriangle.LOWER ? 1 : 0)
					.putInt(16, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(20, diagonal == MatrixDiagonal.UNIT ? 1 : 0)
					.putInt(24, rows).putInt(28, columns);
			if (blasTrsmKernel == null) blasTrsmKernel = new Kernel(BLAS_TRSM_SHADER, 2, 32);
			int count = side == MatrixSide.LEFT ? columns : rows;
			execute(blasTrsmKernel, resources(a, b), push, groups(count, VECTOR_LOCAL_SIZE), 1, 1);
			double[] updated = b.readDoubles(right.length); System.arraycopy(updated, 0, right, 0, right.length);
		} finally { b.close(); a.close(); }
	}

	@Override public synchronized void dcsrmv(double alpha, CsrMatrix matrix, double[] x,
			double beta, double[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		BufferResource values = create(matrix.values());
		BufferResource indices = create(asDoubles(matrix.columnIndices())), starts = create(asDoubles(matrix.rowStarts()));
		BufferResource bx = create(x), by = create(y);
		try {
			ByteBuffer push = nativeBuffer(24).putDouble(0, alpha).putDouble(8, beta).putInt(16, matrix.rows());
			if (csrMvKernel == null) csrMvKernel = new Kernel(CSR_MV_SHADER, 5, 24);
			execute(csrMvKernel, resources(values, indices, starts, bx, by), push,
					groups(matrix.rows(), VECTOR_LOCAL_SIZE), 1, 1);
			double[] updated = by.readDoubles(y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { by.close(); bx.close(); starts.close(); indices.close(); values.close(); }
	}

	@Override public synchronized void dcsrmm(double alpha, CsrMatrix matrix, double[] right,
			int rightColumns, double beta, double[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		BufferResource values = create(matrix.values());
		BufferResource indices = create(asDoubles(matrix.columnIndices())), starts = create(asDoubles(matrix.rowStarts()));
		BufferResource b = create(right), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(24).putDouble(0, alpha).putDouble(8, beta)
					.putInt(16, matrix.rows()).putInt(20, rightColumns);
			if (csrMmKernel == null) csrMmKernel = new Kernel(CSR_MM_SHADER, 5, 24);
			execute(csrMmKernel, resources(values, indices, starts, b, c), push,
					groups(result.length, VECTOR_LOCAL_SIZE), 1, 1);
			double[] updated = c.readDoubles(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); b.close(); starts.close(); indices.close(); values.close(); }
	}

	@Override public synchronized void saxpy(int count, float alpha, float[] x,
			int xOffset, int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return; ensureAvailable(); BufferResource bx = create(x), by = create(y);
		try {
			ByteBuffer push = nativeBuffer(24).putInt(0, count).putFloat(4, alpha)
					.putInt(8, xOffset).putInt(12, xStride).putInt(16, yOffset).putInt(20, yStride);
			if (floatAxpyKernel == null) floatAxpyKernel = new Kernel(FLOAT_AXPY_SHADER, 2, 24);
			execute(floatAxpyKernel, resources(bx, by), push, groups(count, VECTOR_LOCAL_SIZE), 1, 1);
			float[] updated = by.readFloats(y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { by.close(); bx.close(); }
	}

	@Override public synchronized float sdot(int count, float[] x, int xOffset,
			int xStride, float[] y, int yOffset, int yStride) {
		checkRegion(count, x, xOffset, xStride); checkRegion(count, y, yOffset, yStride);
		if (count == 0) return 0.0f; ensureAvailable();
		BufferResource bx = create(x), by = create(y), out = createFloats(1);
		try {
			ByteBuffer push = nativeBuffer(20).putInt(0, count).putInt(4, xOffset)
					.putInt(8, xStride).putInt(12, yOffset).putInt(16, yStride);
			if (floatDotKernel == null) floatDotKernel = new Kernel(FLOAT_DOT_SHADER, 3, 20);
			execute(floatDotKernel, resources(bx, by, out), push, 1, 1, 1);
			return out.readFloats(1)[0];
		} finally { out.close(); by.close(); bx.close(); }
	}

	@Override public synchronized float snrm2(int count, float[] x, int offset, int stride) {
		checkRegion(count, x, offset, stride); if (count == 0) return 0.0f; ensureAvailable();
		BufferResource input = create(x), output = createFloats(1);
		try {
			ByteBuffer push = nativeBuffer(12).putInt(0, count).putInt(4, offset).putInt(8, stride);
			if (floatNrm2Kernel == null) floatNrm2Kernel = new Kernel(FLOAT_NRM2_SHADER, 2, 12);
			execute(floatNrm2Kernel, resources(input, output), push, 1, 1, 1); return output.readFloats(1)[0];
		} finally { output.close(); input.close(); }
	}

	@Override public synchronized void sgemv(MatrixTranspose transpose, int rows, int columns,
			float alpha, float[] matrix, float[] x, float beta, float[] y) {
		checkGemv(transpose, rows, columns, matrix, x, y); ensureAvailable();
		BufferResource a = create(matrix), bx = create(x), by = create(y);
		try {
			ByteBuffer push = nativeBuffer(20).putFloat(0, alpha).putFloat(4, beta)
					.putInt(8, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0).putInt(12, rows).putInt(16, columns);
			if (floatGemvKernel == null) floatGemvKernel = new Kernel(FLOAT_GEMV_SHADER, 3, 20);
			execute(floatGemvKernel, resources(a, bx, by), push, groups(y.length, VECTOR_LOCAL_SIZE), 1, 1);
			float[] updated = by.readFloats(y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { by.close(); bx.close(); a.close(); }
	}

	@Override public synchronized void sgemm(MatrixTranspose leftTranspose, MatrixTranspose rightTranspose,
			int rows, int columns, int shared, float alpha, float[] left, float[] right,
			float beta, float[] result) {
		checkGemm(leftTranspose, rightTranspose, rows, columns, shared, left, right, result); ensureAvailable();
		BufferResource a = create(left), b = create(right), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(28).putFloat(0, alpha).putFloat(4, beta)
					.putInt(8, leftTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(12, rightTranspose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(16, rows).putInt(20, columns).putInt(24, shared);
			if (floatGemmKernel == null) floatGemmKernel = new Kernel(FLOAT_GEMM_SHADER, 3, 28);
			execute(floatGemmKernel, resources(a, b, c), push, groups(columns, 16), groups(rows, 16), 1);
			float[] updated = c.readFloats(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); b.close(); a.close(); }
	}
	@Override public synchronized void ssyrk(MatrixTranspose transpose, int dimension, int shared,
			float alpha, float[] matrix, float beta, float[] result) {
		checkSyrk(transpose, dimension, shared, matrix, result); ensureAvailable();
		BufferResource a = create(matrix), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(20).putFloat(0, alpha).putFloat(4, beta)
					.putInt(8, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(12, dimension).putInt(16, shared);
			if (floatSyrkKernel == null) floatSyrkKernel = new Kernel(FLOAT_SYRK_SHADER, 2, 20);
			execute(floatSyrkKernel, resources(a, c), push, groups(dimension, 16), groups(dimension, 16), 1);
			float[] updated = c.readFloats(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); a.close(); }
	}
	@Override public synchronized void strsv(MatrixTriangle triangle, MatrixTranspose transpose,
			MatrixDiagonal diagonal, int dimension, float[] matrix, float[] vector) {
		checkTriangular(triangle, transpose, diagonal, dimension, matrix, vector); ensureAvailable();
		BufferResource a = create(matrix), x = create(vector);
		try {
			ByteBuffer push = nativeBuffer(16).putInt(0, triangle == MatrixTriangle.LOWER ? 1 : 0)
					.putInt(4, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(8, diagonal == MatrixDiagonal.UNIT ? 1 : 0).putInt(12, dimension);
			if (floatTrsvKernel == null) floatTrsvKernel = new Kernel(FLOAT_TRSV_SHADER, 2, 16);
			execute(floatTrsvKernel, resources(a, x), push, 1, 1, 1);
			float[] updated = x.readFloats(vector.length); System.arraycopy(updated, 0, vector, 0, vector.length);
		} finally { x.close(); a.close(); }
	}
	@Override public synchronized void strsm(MatrixSide side, MatrixTriangle triangle,
			MatrixTranspose transpose, MatrixDiagonal diagonal, int rows, int columns,
			float alpha, float[] matrix, float[] right) {
		checkTrsm(side, triangle, transpose, diagonal, rows, columns, matrix, right); ensureAvailable();
		BufferResource a = create(matrix), b = create(right);
		try {
			ByteBuffer push = nativeBuffer(28).putFloat(0, alpha)
					.putInt(4, side == MatrixSide.RIGHT ? 1 : 0)
					.putInt(8, triangle == MatrixTriangle.LOWER ? 1 : 0)
					.putInt(12, transpose == MatrixTranspose.TRANSPOSE ? 1 : 0)
					.putInt(16, diagonal == MatrixDiagonal.UNIT ? 1 : 0)
					.putInt(20, rows).putInt(24, columns);
			if (floatTrsmKernel == null) floatTrsmKernel = new Kernel(FLOAT_TRSM_SHADER, 2, 28);
			int count = side == MatrixSide.LEFT ? columns : rows;
			execute(floatTrsmKernel, resources(a, b), push, groups(count, VECTOR_LOCAL_SIZE), 1, 1);
			float[] updated = b.readFloats(right.length); System.arraycopy(updated, 0, right, 0, right.length);
		} finally { b.close(); a.close(); }
	}

	@Override public synchronized void scsrmv(float alpha, FloatCsrMatrix matrix, float[] x,
			float beta, float[] y) {
		checkCsrMv(matrix, x, y); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < y.length; i++) y[i] *= beta; return; }
		BufferResource values = create(matrix.values());
		BufferResource indices = create(matrix.columnIndices()), starts = create(matrix.rowStarts());
		BufferResource bx = create(x), by = create(y);
		try {
			ByteBuffer push = nativeBuffer(12).putFloat(0, alpha).putFloat(4, beta).putInt(8, matrix.rows());
			if (floatCsrMvKernel == null) floatCsrMvKernel = new Kernel(FLOAT_CSR_MV_SHADER, 5, 12);
			execute(floatCsrMvKernel, resources(values, indices, starts, bx, by), push,
					groups(matrix.rows(), VECTOR_LOCAL_SIZE), 1, 1);
			float[] updated = by.readFloats(y.length); System.arraycopy(updated, 0, y, 0, y.length);
		} finally { by.close(); bx.close(); starts.close(); indices.close(); values.close(); }
	}

	@Override public synchronized void scsrmm(float alpha, FloatCsrMatrix matrix, float[] right,
			int rightColumns, float beta, float[] result) {
		checkCsrMm(matrix, right, rightColumns, result); ensureAvailable();
		if (matrix.nonzeroCount() == 0) { for (int i = 0; i < result.length; i++) result[i] *= beta; return; }
		BufferResource values = create(matrix.values());
		BufferResource indices = create(matrix.columnIndices()), starts = create(matrix.rowStarts());
		BufferResource b = create(right), c = create(result);
		try {
			ByteBuffer push = nativeBuffer(16).putFloat(0, alpha).putFloat(4, beta)
					.putInt(8, matrix.rows()).putInt(12, rightColumns);
			if (floatCsrMmKernel == null) floatCsrMmKernel = new Kernel(FLOAT_CSR_MM_SHADER, 5, 16);
			execute(floatCsrMmKernel, resources(values, indices, starts, b, c), push,
					groups(result.length, VECTOR_LOCAL_SIZE), 1, 1);
			float[] updated = c.readFloats(result.length); System.arraycopy(updated, 0, result, 0, result.length);
		} finally { c.close(); b.close(); starts.close(); indices.close(); values.close(); }
	}

	@Override public synchronized CholeskyFactor dpotrf(double[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);ensureAvailable();BufferResource a=create(matrix),lower=createDoubles(matrix.length),info=createInts(1);try{ByteBuffer push=nativeBuffer(4).putInt(0,dimension);if(dpotrfKernel==null)dpotrfKernel=new Kernel(DPOTRF_SHADER,3,4);execute(dpotrfKernel,resources(a,lower,info),push,1,1,1);int status=info.readInts(1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);return new CholeskyFactor(dimension,lower.readDoubles(matrix.length));}finally{info.close();lower.close();a.close();}}
	@Override public synchronized PivotedQrFactor dgeqp3(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);BufferResource qr=create(matrix),tau=createDoubles(count),pivot=createInts(columns);try{ByteBuffer push=nativeBuffer(8).putInt(0,rows).putInt(4,columns);if(dgeqp3Kernel==null)dgeqp3Kernel=new Kernel(DGEQP3_SHADER,3,8);execute(dgeqp3Kernel,resources(qr,tau,pivot),push,1,1,1);return new PivotedQrFactor(rows,columns,qr.readDoubles(matrix.length),tau.readDoubles(count),pivot.readInts(columns));}finally{pivot.close();tau.close();qr.close();}}
	@Override public synchronized SymmetricEigenDecomposition dsyev(double[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();BufferResource work=create(matrix),values=createDoubles(dimension),vectors=createDoubles(matrix.length),info=createInts(1);try{ByteBuffer push=nativeBuffer(16).putDouble(0,16*Math.ulp(1.0)).putInt(8,dimension);if(dsyevKernel==null)dsyevKernel=new Kernel(DSYEV_SHADER,4,16);execute(dsyevKernel,resources(work,values,vectors,info),push,1,1,1);if(info.readInts(1)[0]!=0)throw new IllegalStateException("Vulkan symmetric eigendecomposition did not converge");return new SymmetricEigenDecomposition(dimension,values.readDoubles(dimension),vectors.readDoubles(matrix.length));}finally{info.close();vectors.close();values.close();work.close();}}
	@Override public synchronized SingularValueDecomposition dgesvd(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);BufferResource a=create(matrix),work=createDoubles(matrix.length),u=createDoubles(rows*count),singular=createDoubles(count),vt=createDoubles(count*columns),info=createInts(1);try{ByteBuffer push=nativeBuffer(16).putDouble(0,16*Math.ulp(1.0)).putInt(8,rows).putInt(12,columns);if(dgesvdKernel==null)dgesvdKernel=new Kernel(DGESVD_SHADER,6,16);execute(dgesvdKernel,resources(a,work,u,singular,vt,info),push,1,1,1);if(info.readInts(1)[0]!=0)throw new IllegalStateException("Vulkan SVD did not converge");return new SingularValueDecomposition(rows,columns,singular.readDoubles(count),u.readDoubles(rows*count),vt.readDoubles(count*columns));}finally{info.close();vt.close();singular.close();u.close();work.close();a.close();}}
	@Override public synchronized FloatCholeskyFactor spotrf(float[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);ensureAvailable();BufferResource a=create(matrix),lower=createFloats(matrix.length),info=createInts(1);try{ByteBuffer push=nativeBuffer(4).putInt(0,dimension);if(spotrfKernel==null)spotrfKernel=new Kernel(SPOTRF_SHADER,3,4);execute(spotrfKernel,resources(a,lower,info),push,1,1,1);int status=info.readInts(1)[0];if(status!=0)throw new IllegalArgumentException("matrix is not positive definite at minor "+status);return new FloatCholeskyFactor(dimension,lower.readFloats(matrix.length));}finally{info.close();lower.close();a.close();}}
	@Override public synchronized FloatPivotedQrFactor sgeqp3(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);BufferResource qr=create(matrix),tau=createFloats(count),pivot=createInts(columns);try{ByteBuffer push=nativeBuffer(8).putInt(0,rows).putInt(4,columns);if(sgeqp3Kernel==null)sgeqp3Kernel=new Kernel(SGEQP3_SHADER,3,8);execute(sgeqp3Kernel,resources(qr,tau,pivot),push,1,1,1);return new FloatPivotedQrFactor(rows,columns,qr.readFloats(matrix.length),tau.readFloats(count),pivot.readInts(columns));}finally{pivot.close();tau.close();qr.close();}}
	@Override public synchronized FloatSymmetricEigenDecomposition ssyev(float[]matrix,int dimension){checkDecompositionMatrix(matrix,dimension,dimension);checkSymmetric(matrix,dimension);ensureAvailable();BufferResource work=create(matrix),values=createFloats(dimension),vectors=createFloats(matrix.length),info=createInts(1);try{ByteBuffer push=nativeBuffer(8).putFloat(0,16*Math.ulp(1.0f)).putInt(4,dimension);if(ssyevKernel==null)ssyevKernel=new Kernel(SSYEV_SHADER,4,8);execute(ssyevKernel,resources(work,values,vectors,info),push,1,1,1);if(info.readInts(1)[0]!=0)throw new IllegalStateException("Vulkan FP32 symmetric eigendecomposition did not converge");return new FloatSymmetricEigenDecomposition(dimension,values.readFloats(dimension),vectors.readFloats(matrix.length));}finally{info.close();vectors.close();values.close();work.close();}}
	@Override public synchronized FloatSingularValueDecomposition sgesvd(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();int count=Math.min(rows,columns);BufferResource a=create(matrix),work=createFloats(matrix.length),u=createFloats(rows*count),singular=createFloats(count),vt=createFloats(count*columns),info=createInts(1);try{ByteBuffer push=nativeBuffer(12).putFloat(0,16*Math.ulp(1.0f)).putInt(4,rows).putInt(8,columns);if(sgesvdKernel==null)sgesvdKernel=new Kernel(SGESVD_SHADER,6,12);execute(sgesvdKernel,resources(a,work,u,singular,vt,info),push,1,1,1);if(info.readInts(1)[0]!=0)throw new IllegalStateException("Vulkan FP32 SVD did not converge");return new FloatSingularValueDecomposition(rows,columns,singular.readFloats(count),u.readFloats(rows*count),vt.readFloats(count*columns));}finally{info.close();vt.close();singular.close();u.close();work.close();a.close();}}
	@Override public synchronized PreparedDenseMatrix prepareDge(double[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();return new PreparedDoubleDense(matrix,rows,columns);}
	@Override public synchronized PreparedFloatDenseMatrix prepareSge(float[]matrix,int rows,int columns){checkDecompositionMatrix(matrix,rows,columns);ensureAvailable();return new PreparedFloatDense(matrix,rows,columns);}
	private final class PreparedDoubleDense implements PreparedDenseMatrix{
		private final int rows,columns;private BufferResource matrix;
		PreparedDoubleDense(double[]source,int rows,int columns){this.rows=rows;this.columns=columns;matrix=create(source);}public int rows(){checkOpen();return rows;}public int columns(){checkOpen();return columns;}
		public void multiply(MatrixTranspose transpose,double alpha,double[]right,int rightColumns,double beta,double[]result){if(transpose==null||rightColumns<1)throw new IllegalArgumentException("prepared Vulkan dense dimensions do not conform");int output=transpose==MatrixTranspose.NONE?rows:columns,shared=transpose==MatrixTranspose.NONE?columns:rows;if(right==null||right.length!=shared*rightColumns||result==null||result.length!=output*rightColumns)throw new IllegalArgumentException("prepared Vulkan dense dimensions do not conform");synchronized(VulkanComputeBackend.this){checkOpen();BufferResource b=create(right),c=create(result);try{ByteBuffer push=nativeBuffer(48).putDouble(0,alpha).putDouble(8,beta).putInt(16,transpose==MatrixTranspose.TRANSPOSE?1:0).putInt(20,0).putInt(24,output).putInt(28,rightColumns).putInt(32,shared);if(blasGemmKernel==null)blasGemmKernel=new Kernel(BLAS_GEMM_SHADER,3,48);execute(blasGemmKernel,resources(matrix,b,c),push,groups(rightColumns,16),groups(output,16),1);double[]updated=c.readDoubles(result.length);System.arraycopy(updated,0,result,0,result.length);}finally{c.close();b.close();}}}
		public void close(){synchronized(VulkanComputeBackend.this){if(matrix!=null){matrix.close();matrix=null;}}}private void checkOpen(){if(matrix==null)throw new IllegalStateException("prepared Vulkan dense matrix is closed");}
	}
	private final class PreparedFloatDense implements PreparedFloatDenseMatrix{
		private final int rows,columns;private BufferResource matrix;
		PreparedFloatDense(float[]source,int rows,int columns){this.rows=rows;this.columns=columns;matrix=create(source);}public int rows(){checkOpen();return rows;}public int columns(){checkOpen();return columns;}
		public void multiply(MatrixTranspose transpose,float alpha,float[]right,int rightColumns,float beta,float[]result){if(transpose==null||rightColumns<1)throw new IllegalArgumentException("prepared Vulkan FP32 dense dimensions do not conform");int output=transpose==MatrixTranspose.NONE?rows:columns,shared=transpose==MatrixTranspose.NONE?columns:rows;if(right==null||right.length!=shared*rightColumns||result==null||result.length!=output*rightColumns)throw new IllegalArgumentException("prepared Vulkan FP32 dense dimensions do not conform");synchronized(VulkanComputeBackend.this){checkOpen();BufferResource b=create(right),c=create(result);try{ByteBuffer push=nativeBuffer(28).putFloat(0,alpha).putFloat(4,beta).putInt(8,transpose==MatrixTranspose.TRANSPOSE?1:0).putInt(12,0).putInt(16,output).putInt(20,rightColumns).putInt(24,shared);if(floatGemmKernel==null)floatGemmKernel=new Kernel(FLOAT_GEMM_SHADER,3,28);execute(floatGemmKernel,resources(matrix,b,c),push,groups(rightColumns,16),groups(output,16),1);float[]updated=c.readFloats(result.length);System.arraycopy(updated,0,result,0,result.length);}finally{c.close();b.close();}}}
		public void close(){synchronized(VulkanComputeBackend.this){if(matrix!=null){matrix.close();matrix=null;}}}private void checkOpen(){if(matrix==null)throw new IllegalStateException("prepared Vulkan FP32 dense matrix is closed");}
	}

	@Override public synchronized LogisticRegressionBatchResult logisticRegression(double[][] design,
			double[] outcomes, double[][] states, double priorPrecision) {
		int[] xShape = shape(design), qShape = shape(states);
		if (outcomes == null || outcomes.length != xShape[0] || qShape[1] != xShape[1]
				|| !(priorPrecision >= 0.0)) throw new IllegalArgumentException("invalid logistic batch");
		ensureAvailable(); int chains = qShape[0], dimensions = qShape[1];
		BufferResource x = create(flatten(design)), y = create(outcomes), q = create(flatten(states));
		BufferResource values = createDoubles(chains), gradients = createDoubles(chains * dimensions);
		try {
			ByteBuffer push = nativeBuffer(24).putDouble(0, priorPrecision).putInt(8, xShape[0])
					.putInt(12, dimensions).putInt(16, chains);
			if (logisticKernel == null) logisticKernel = new Kernel(LOGISTIC_SHADER, 5, 24);
			execute(logisticKernel, resources(x, y, q, values, gradients), push, groups(chains, 64), 1, 1);
			return new LogisticRegressionBatchResult(values.readDoubles(chains),
					reshape(gradients.readDoubles(chains * dimensions), chains, dimensions));
		} finally { gradients.close(); values.close(); q.close(); y.close(); x.close(); }
	}

	@Override public PreparedLogisticRegression prepareLogisticRegression(double[][] design,
			double[] outcomes) {
		int[] shape = shape(design);
		if (outcomes == null || outcomes.length != shape[0])
			throw new IllegalArgumentException("one outcome per row is required");
		final double[][] copiedDesign = new double[shape[0]][];
		for (int i = 0; i < shape[0]; i++) copiedDesign[i] = design[i].clone();
		final double[] copiedOutcomes = outcomes.clone();
		return new PreparedLogisticRegression() {
			@Override public int rows() { return copiedDesign.length; }
			@Override public int dimensions() { return copiedDesign[0].length; }
			@Override public LogisticRegressionBatchResult evaluate(double[][] states, double priorPrecision) {
				return logisticRegression(copiedDesign, copiedOutcomes, states, priorPrecision);
			}
			@Override public void close() {}
		};
	}

	private void initialize() {
		try (MemoryStack stack = stackPush()) {
			VkApplicationInfo application = VkApplicationInfo.calloc(stack).sType$Default()
					.pApplicationName(stack.UTF8("JDistlib")).applicationVersion(VK_MAKE_VERSION(0, 8, 4))
					.pEngineName(stack.UTF8("JDistlib")).engineVersion(VK_MAKE_VERSION(0, 8, 4))
					.apiVersion(VK_API_VERSION_1_0);
			VkInstanceCreateInfo create = VkInstanceCreateInfo.calloc(stack).sType$Default().pApplicationInfo(application);
			PointerBuffer pointer = stack.mallocPointer(1);
			check(vkCreateInstance(create, null, pointer), "create Vulkan instance");
			instance = new VkInstance(pointer.get(0), create);

			IntBuffer count = stack.ints(0);
			check(vkEnumeratePhysicalDevices(instance, count, null), "enumerate Vulkan devices");
			if (count.get(0) == 0) throw new IllegalStateException("no Vulkan physical device");
			PointerBuffer devices = stack.mallocPointer(count.get(0));
			check(vkEnumeratePhysicalDevices(instance, count, devices), "enumerate Vulkan devices");
			for (int i = 0; i < devices.capacity() && physicalDevice == null; i++) {
				VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
				VkPhysicalDeviceFeatures features = VkPhysicalDeviceFeatures.calloc(stack);
				vkGetPhysicalDeviceFeatures(candidate, features);
				if (!features.shaderFloat64()) continue;
				int family = computeQueueFamily(candidate, stack);
				if (family >= 0) { physicalDevice = candidate; queueFamily = family; }
			}
			if (physicalDevice == null) throw new IllegalStateException("no compute-capable Vulkan GPU with shaderFloat64");

			VkDeviceQueueCreateInfo.Buffer queueCreate = VkDeviceQueueCreateInfo.calloc(1, stack);
			queueCreate.get(0).sType$Default().queueFamilyIndex(queueFamily).pQueuePriorities(stack.floats(1.0f));
			VkPhysicalDeviceFeatures enabled = VkPhysicalDeviceFeatures.calloc(stack).shaderFloat64(true);
			VkDeviceCreateInfo deviceCreate = VkDeviceCreateInfo.calloc(stack).sType$Default()
					.pQueueCreateInfos(queueCreate).pEnabledFeatures(enabled);
			check(vkCreateDevice(physicalDevice, deviceCreate, null, pointer), "create Vulkan device");
			device = new VkDevice(pointer.get(0), physicalDevice, deviceCreate);
			vkGetDeviceQueue(device, queueFamily, 0, pointer);
			queue = new VkQueue(pointer.get(0), device);

			LongBuffer pool = stack.mallocLong(1);
			VkCommandPoolCreateInfo poolCreate = VkCommandPoolCreateInfo.calloc(stack).sType$Default()
					.queueFamilyIndex(queueFamily).flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
			check(vkCreateCommandPool(device, poolCreate, null, pool), "create Vulkan command pool");
			commandPool = pool.get(0);

			VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.calloc(stack);
			vkGetPhysicalDeviceProperties(physicalDevice, properties);
			capabilities = new ComputeCapabilities("VULKAN", properties.deviceNameString(), true, true,
					deviceLocalMemory(physicalDevice, stack), true, true, true,
					false, false, true, true, true);
			Package pkg = getClass().getPackage(); String backendVersion = pkg == null
					|| pkg.getImplementationVersion() == null ? "development" : pkg.getImplementationVersion();
			deviceInfo = new ComputeDeviceInfo(id(), backendVersion, ComputeApi.VULKAN,
					vulkanVersion(properties.apiVersion()), Integer.toUnsignedString(properties.driverVersion()),
					vendor(properties.vendorID()), properties.deviceNameString(),
					deviceType(properties.deviceType()), Integer.toHexString(properties.deviceID()),
					capabilities.globalMemoryBytes());
		}
	}
	private static String vulkanVersion(int version) {
		return (version >>> 22) + "." + ((version >>> 12) & 0x3ff) + "." + (version & 0xfff);
	}
	private static String vendor(int id) {
		switch (id) { case 0x10de: return "NVIDIA"; case 0x1002: return "AMD";
		case 0x8086: return "Intel"; default: return "0x" + Integer.toHexString(id); }
	}
	private static String deviceType(int type) {
		switch (type) { case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
		case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU"; default: return "other"; }
	}

	private int computeQueueFamily(VkPhysicalDevice candidate, MemoryStack stack) {
		IntBuffer count = stack.ints(0);
		vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, null);
		VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(count.get(0), stack);
		vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, families);
		for (int i = 0; i < families.capacity(); i++)
			if ((families.get(i).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) return i;
		return -1;
	}

	private long deviceLocalMemory(VkPhysicalDevice candidate, MemoryStack stack) {
		VkPhysicalDeviceMemoryProperties memory = VkPhysicalDeviceMemoryProperties.calloc(stack);
		vkGetPhysicalDeviceMemoryProperties(candidate, memory); long total = 0L;
		for (int i = 0; i < memory.memoryHeapCount(); i++) {
			VkMemoryHeap heap = memory.memoryHeaps(i);
			if ((heap.flags() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) total = saturatingAdd(total, heap.size());
		}
		return total;
	}

	private BufferResource create(double[] values) {
		BufferResource resource = createDoubles(values.length); resource.write(values); return resource;
	}
	private BufferResource create(float[] values) {
		BufferResource resource = createFloats(values.length); resource.write(values); return resource;
	}
	private BufferResource create(int[] values) {
		if (values.length <= 0) throw new IllegalArgumentException("buffer must be nonempty");
		BufferResource resource = new BufferResource(Math.multiplyExact((long) values.length, 4L));
		resource.write(values); return resource;
	}

	private BufferResource createDoubles(int length) {
		if (length <= 0) throw new IllegalArgumentException("buffer must be nonempty");
		return new BufferResource(Math.multiplyExact((long) length, 8L));
	}
	private BufferResource createFloats(int length) {
		if (length <= 0) throw new IllegalArgumentException("buffer must be nonempty");
		return new BufferResource(Math.multiplyExact((long) length, 4L));
	}
	private BufferResource createInts(int length) {
		if (length <= 0) throw new IllegalArgumentException("buffer must be nonempty");
		return new BufferResource(Math.multiplyExact((long) length, 4L));
	}

	private final class BufferResource implements AutoCloseable {
		private final long size;
		private long buffer, memory;
		BufferResource(long size) {
			this.size = size;
			try (MemoryStack stack = stackPush()) {
				LongBuffer value = stack.mallocLong(1);
				VkBufferCreateInfo create = VkBufferCreateInfo.calloc(stack).sType$Default().size(size)
						.usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT).sharingMode(VK_SHARING_MODE_EXCLUSIVE);
				check(vkCreateBuffer(device, create, null, value), "create Vulkan storage buffer"); buffer = value.get(0);
				VkMemoryRequirements requirements = VkMemoryRequirements.calloc(stack);
				vkGetBufferMemoryRequirements(device, buffer, requirements);
				VkMemoryAllocateInfo allocation = VkMemoryAllocateInfo.calloc(stack).sType$Default()
						.allocationSize(requirements.size()).memoryTypeIndex(findMemoryType(requirements.memoryTypeBits(), stack));
				check(vkAllocateMemory(device, allocation, null, value), "allocate Vulkan buffer memory"); memory = value.get(0);
				check(vkBindBufferMemory(device, buffer, memory, 0L), "bind Vulkan buffer memory");
			}
		}
		void write(double[] values) {
			if ((long) values.length * 8L > size) throw new IllegalArgumentException("buffer overflow");
			ByteBuffer mapped = map(); try { mapped.asDoubleBuffer().put(values); } finally { unmap(); }
		}
		void write(float[] values) {
			if ((long) values.length * 4L > size) throw new IllegalArgumentException("buffer overflow");
			ByteBuffer mapped = map(); try { mapped.asFloatBuffer().put(values); } finally { unmap(); }
		}
		void write(int[] values) {
			if ((long) values.length * 4L > size) throw new IllegalArgumentException("buffer overflow");
			ByteBuffer mapped = map(); try { mapped.asIntBuffer().put(values); } finally { unmap(); }
		}
		double[] readDoubles(int count) {
			if ((long) count * 8L > size) throw new IllegalArgumentException("buffer overflow");
			double[] values = new double[count]; ByteBuffer mapped = map();
			try { mapped.asDoubleBuffer().get(values); } finally { unmap(); } return values;
		}
		float[] readFloats(int count) {
			if ((long) count * 4L > size) throw new IllegalArgumentException("buffer overflow");
			float[] values = new float[count]; ByteBuffer mapped = map();
			try { mapped.asFloatBuffer().get(values); } finally { unmap(); } return values;
		}
		int[] readInts(int count) {
			if ((long) count * 4L > size) throw new IllegalArgumentException("buffer overflow");
			int[] values = new int[count]; ByteBuffer mapped = map();
			try { mapped.asIntBuffer().get(values); } finally { unmap(); } return values;
		}
		private ByteBuffer map() {
			try (MemoryStack stack = stackPush()) {
				PointerBuffer pointer = stack.mallocPointer(1);
				check(vkMapMemory(device, memory, 0L, size, 0, pointer), "map Vulkan buffer memory");
				return memByteBuffer(pointer.get(0), Math.toIntExact(size)).order(ByteOrder.nativeOrder());
			}
		}
		private void unmap() { vkUnmapMemory(device, memory); }
		@Override public void close() {
			if (buffer != NULL && device != null) vkDestroyBuffer(device, buffer, null);
			if (memory != NULL && device != null) vkFreeMemory(device, memory, null);
			buffer = memory = NULL;
		}
	}

	private int findMemoryType(int allowed, MemoryStack stack) {
		VkPhysicalDeviceMemoryProperties properties = VkPhysicalDeviceMemoryProperties.calloc(stack);
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
		int required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		for (int i = 0; i < properties.memoryTypeCount(); i++)
			if ((allowed & (1 << i)) != 0 && (properties.memoryTypes(i).propertyFlags() & required) == required) return i;
		throw new IllegalStateException("Vulkan device has no host-visible coherent storage memory");
	}

	private void execute(Kernel kernel, List<BufferResource> buffers, ByteBuffer push,
			int groupX, int groupY, int groupZ) {
		ensureAvailable();
		if (buffers.size() != kernel.bufferCount || push == null || push.capacity() != kernel.pushSize)
			throw new IllegalArgumentException("Vulkan kernel argument layout mismatch");
		long descriptorPool = NULL, fence = NULL; VkCommandBuffer command = null;
		try (MemoryStack stack = stackPush()) {
			LongBuffer handle = stack.mallocLong(1);
			VkDescriptorPoolSize.Buffer poolSize = VkDescriptorPoolSize.calloc(1, stack);
			poolSize.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(buffers.size());
			VkDescriptorPoolCreateInfo poolCreate = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
					.maxSets(1).pPoolSizes(poolSize);
			check(vkCreateDescriptorPool(device, poolCreate, null, handle), "create Vulkan descriptor pool");
			descriptorPool = handle.get(0);
			VkDescriptorSetAllocateInfo setAllocate = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
					.descriptorPool(descriptorPool).pSetLayouts(stack.longs(kernel.descriptorLayout));
			check(vkAllocateDescriptorSets(device, setAllocate, handle), "allocate Vulkan descriptor set");
			long descriptorSet = handle.get(0);

			VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(buffers.size(), stack);
			for (int i = 0; i < buffers.size(); i++) {
				VkDescriptorBufferInfo.Buffer info = VkDescriptorBufferInfo.calloc(1, stack);
				info.get(0).buffer(buffers.get(i).buffer).offset(0L).range(buffers.get(i).size);
				writes.get(i).sType$Default().dstSet(descriptorSet).dstBinding(i)
						.descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(info);
			}
			vkUpdateDescriptorSets(device, writes, null);

			VkCommandBufferAllocateInfo commandAllocate = VkCommandBufferAllocateInfo.calloc(stack).sType$Default()
					.commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
			PointerBuffer commandPointer = stack.mallocPointer(1);
			check(vkAllocateCommandBuffers(device, commandAllocate, commandPointer), "allocate Vulkan command buffer");
			command = new VkCommandBuffer(commandPointer.get(0), device);
			VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack).sType$Default()
					.flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			check(vkBeginCommandBuffer(command, begin), "begin Vulkan command buffer");
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0,
					stack.longs(descriptorSet), null);
			if (push.capacity() > 0) vkCmdPushConstants(command, kernel.pipelineLayout,
					VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
			vkCmdDispatch(command, groupX, groupY, groupZ);
			check(vkEndCommandBuffer(command), "end Vulkan command buffer");

			VkFenceCreateInfo fenceCreate = VkFenceCreateInfo.calloc(stack).sType$Default();
			check(vkCreateFence(device, fenceCreate, null, handle), "create Vulkan fence"); fence = handle.get(0);
			VkSubmitInfo.Buffer submit = VkSubmitInfo.calloc(1, stack);
			submit.get(0).sType$Default().pCommandBuffers(stack.pointers(command.address()));
			check(vkQueueSubmit(queue, submit, fence), "submit Vulkan compute work");
			check(vkWaitForFences(device, fence, true, Long.MAX_VALUE), "wait for Vulkan compute work");
		} finally {
			if (device != null) {
				if (fence != NULL) vkDestroyFence(device, fence, null);
				if (command != null) vkFreeCommandBuffers(device, commandPool, command);
				if (descriptorPool != NULL) vkDestroyDescriptorPool(device, descriptorPool, null);
			}
		}
	}

	private final class Kernel implements AutoCloseable {
		private final int bufferCount, pushSize;
		private long shaderModule, descriptorLayout, pipelineLayout, pipeline;
		Kernel(String shader, int bufferCount, int pushSize) {
			this.bufferCount = bufferCount; this.pushSize = pushSize;
			try (MemoryStack stack = stackPush()) {
				LongBuffer handle = stack.mallocLong(1);
				shaderModule = createShaderModule(shader, stack);
				VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(bufferCount, stack);
				for (int i = 0; i < bufferCount; i++) bindings.get(i).binding(i)
						.descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
						.stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
				VkDescriptorSetLayoutCreateInfo descriptorCreate = VkDescriptorSetLayoutCreateInfo.calloc(stack)
						.sType$Default().pBindings(bindings);
				check(vkCreateDescriptorSetLayout(device, descriptorCreate, null, handle),
						"create Vulkan descriptor layout"); descriptorLayout = handle.get(0);

				VkPipelineLayoutCreateInfo layoutCreate = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
						.pSetLayouts(stack.longs(descriptorLayout));
				VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, stack);
				range.get(0).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushSize);
				layoutCreate.pPushConstantRanges(range);
				check(vkCreatePipelineLayout(device, layoutCreate, null, handle),
						"create Vulkan pipeline layout"); pipelineLayout = handle.get(0);

				VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
						.stage(VK_SHADER_STAGE_COMPUTE_BIT).module(shaderModule).pName(stack.UTF8("main"));
				VkComputePipelineCreateInfo.Buffer pipelineCreate = VkComputePipelineCreateInfo.calloc(1, stack);
				pipelineCreate.get(0).sType$Default().stage(stage).layout(pipelineLayout);
				check(vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineCreate, null, handle),
						"create Vulkan compute pipeline"); pipeline = handle.get(0);
			} catch (RuntimeException error) { close(); throw error; }
			catch (Error error) { close(); throw error; }
		}
		@Override public void close() {
			if (device != null) {
				if (pipeline != NULL) vkDestroyPipeline(device, pipeline, null);
				if (pipelineLayout != NULL) vkDestroyPipelineLayout(device, pipelineLayout, null);
				if (descriptorLayout != NULL) vkDestroyDescriptorSetLayout(device, descriptorLayout, null);
				if (shaderModule != NULL) vkDestroyShaderModule(device, shaderModule, null);
			}
			pipeline = pipelineLayout = descriptorLayout = shaderModule = NULL;
		}
	}

	private long createShaderModule(String source, MemoryStack stack) {
		long compiler = shaderc_compiler_initialize();
		if (compiler == NULL) throw new IllegalStateException("cannot initialize shaderc");
		long result = NULL;
		try {
			result = shaderc_compile_into_spv(compiler, source, shaderc_compute_shader,
					"jdistlib-vulkan.comp", "main", NULL);
			if (result == NULL) throw new IllegalStateException("shaderc returned no result");
			if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
				throw new IllegalStateException("Vulkan shader compilation failed: " + shaderc_result_get_error_message(result));
			ByteBuffer code = shaderc_result_get_bytes(result);
			VkShaderModuleCreateInfo create = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
			LongBuffer module = stack.mallocLong(1);
			check(vkCreateShaderModule(device, create, null, module), "create Vulkan shader module");
			return module.get(0);
		} finally {
			if (result != NULL) shaderc_result_release(result);
			shaderc_compiler_release(compiler);
		}
	}

	@Override public synchronized void close() {
		if (device != null) {
			try { vkDeviceWaitIdle(device); } catch (Throwable ignored) {}
			closeKernel(sgesvdKernel); closeKernel(ssyevKernel); closeKernel(sgeqp3Kernel); closeKernel(spotrfKernel);
			closeKernel(dgesvdKernel); closeKernel(dsyevKernel); closeKernel(dgeqp3Kernel); closeKernel(dpotrfKernel);
			closeKernel(floatCsrMmKernel); closeKernel(floatCsrMvKernel); closeKernel(floatTrsmKernel);
			closeKernel(floatTrsvKernel); closeKernel(floatSyrkKernel); closeKernel(floatGemmKernel);
			closeKernel(floatGemvKernel); closeKernel(floatNrm2Kernel); closeKernel(floatDotKernel);
			closeKernel(floatAxpyKernel);
			closeKernel(csrMmKernel); closeKernel(csrMvKernel); closeKernel(blasTrsmKernel);
			closeKernel(blasTrsvKernel); closeKernel(blasSyrkKernel); closeKernel(blasGemmKernel);
			closeKernel(blasGemvKernel); closeKernel(blasNrm2Kernel);
			closeKernel(logisticKernel); closeKernel(gemmKernel); closeKernel(dotKernel);
			closeKernel(axpyKernel); closeKernel(unaryKernel);
			csrMmKernel = csrMvKernel = blasTrsmKernel = blasTrsvKernel = blasSyrkKernel = null;
			blasGemmKernel = blasGemvKernel = blasNrm2Kernel = null;
			floatCsrMmKernel = floatCsrMvKernel = floatTrsmKernel = floatTrsvKernel = null;
			floatSyrkKernel = floatGemmKernel = null;
			floatCsrMmKernel = floatCsrMvKernel = floatGemmKernel = floatGemvKernel = null;
			floatNrm2Kernel = floatDotKernel = floatAxpyKernel = null;
			sgesvdKernel = ssyevKernel = sgeqp3Kernel = spotrfKernel = null;
			dgesvdKernel = dsyevKernel = dgeqp3Kernel = dpotrfKernel = null;
			logisticKernel = gemmKernel = dotKernel = axpyKernel = unaryKernel = null;
			if (commandPool != NULL) vkDestroyCommandPool(device, commandPool, null);
			vkDestroyDevice(device, null); device = null; queue = null; commandPool = NULL;
		}
		if (instance != null) { vkDestroyInstance(instance, null); instance = null; physicalDevice = null; }
	}
	private static void closeKernel(Kernel kernel) { if (kernel != null) kernel.close(); }

	private void ensureAvailable() {
		if (!available()) throw new IllegalStateException("Vulkan backend is unavailable", unavailableCause);
	}
	private static void check(int status, String action) {
		if (status != VK_SUCCESS) throw new IllegalStateException(action + " failed with Vulkan status " + status);
	}
	private static int groups(int count, int local) { return (count + local - 1) / local; }
	private static ByteBuffer nativeBuffer(int size) {
		return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
	}
	private static List<BufferResource> resources(BufferResource... values) {
		List<BufferResource> result = new ArrayList<BufferResource>(values.length);
		for (BufferResource value : values) result.add(value); return result;
	}
	private static void checkVectors(double[] x, double[] y) {
		if (x == null || y == null || x.length == 0 || x.length != y.length)
			throw new IllegalArgumentException("vector lengths must match and be nonzero");
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
	private static double[] asDoubles(int[] values) {
		double[] result = new double[values.length]; for (int i = 0; i < values.length; i++) result[i] = values[i]; return result;
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
		int[] shape = shape(matrix); double[] values = new double[shape[0] * shape[1]]; int offset = 0;
		for (double[] row : matrix) { System.arraycopy(row, 0, values, offset, row.length); offset += row.length; }
		return values;
	}
	private static double[][] reshape(double[] values, int rows, int columns) {
		double[][] result = new double[rows][columns];
		for (int row = 0; row < rows; row++) System.arraycopy(values, row * columns, result[row], 0, columns);
		return result;
	}
	private static long saturatingAdd(long left, long right) {
		return Long.MAX_VALUE - left < right ? Long.MAX_VALUE : left + right;
	}
}
