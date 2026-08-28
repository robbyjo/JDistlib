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
import jdistlib.accelerator.LogisticRegressionBatchResult;
import jdistlib.accelerator.PreparedLogisticRegression;
import jdistlib.accelerator.UnaryOperation;

/**
 * Optional FP64 Vulkan 1.0 compute backend using LWJGL and runtime GLSL-to-SPIR-V
 * compilation. The provider deliberately uses host-visible storage so it can run
 * across discrete and integrated Vulkan devices without a vendor-specific copy path.
 */
public final class VulkanComputeBackend implements ComputeBackend {
	private static final int VECTOR_LOCAL_SIZE = 256;
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

	private VkInstance instance;
	private VkPhysicalDevice physicalDevice;
	private VkDevice device;
	private VkQueue queue;
	private int queueFamily = -1;
	private long commandPool;
	private Kernel unaryKernel, axpyKernel, dotKernel, gemmKernel, logisticKernel;
	private ComputeCapabilities capabilities;
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
					deviceLocalMemory(physicalDevice, stack));
		}
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

	private BufferResource createDoubles(int length) {
		if (length <= 0) throw new IllegalArgumentException("buffer must be nonempty");
		return new BufferResource(Math.multiplyExact((long) length, 8L));
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
		double[] readDoubles(int count) {
			if ((long) count * 8L > size) throw new IllegalArgumentException("buffer overflow");
			double[] values = new double[count]; ByteBuffer mapped = map();
			try { mapped.asDoubleBuffer().get(values); } finally { unmap(); } return values;
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
			closeKernel(logisticKernel); closeKernel(gemmKernel); closeKernel(dotKernel);
			closeKernel(axpyKernel); closeKernel(unaryKernel);
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
