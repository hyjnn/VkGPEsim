#include <print>

#include <vulkan/vulkan_core.h>
#include <vkFFT/vkFFT.h>
#include <utils_vkFFT/utils_VkFFT.h>

namespace gpe {
	inline VkResult InitGPU(VkGPU &gpu, VkFFTConfiguration &config) {
		VkResult res = VK_SUCCESS;

		res = createInstance(&gpu, 0);
		if (res) {
			std::print("Instace creation failed, error code {}", static_cast<int>(res));
			return res;
		}
		res = setupDebugMessenger(&gpu);
		if (res) {
			std::print("Debug messenger creation failed, error code {}", static_cast<int>(res));
			return res;
		}
		res = findPhysicalDevice(&gpu);
		if (res) {
			std::print("Failed to find physical device, error code {}", static_cast<int>(res));
			return res;
		}
		res = createDevice(&gpu, 0);
		if (res) {
			std::print("Device creation failed, error code {}", static_cast<int>(res));
			return res;
		}
		res = createFence(&gpu);
		if (res) {
			std::print("Device creation failed, error code {}", static_cast<int>(res));
			return res;
		}
		res = createCommandPool(&gpu);
		if (res) {
			std::print("Command pool creation failed, error code {}", static_cast<int>(res));
			return res;
		}

		vkGetPhysicalDeviceProperties(gpu.physicalDevice, &gpu.physicalDeviceProperties);
		vkGetPhysicalDeviceMemoryProperties(gpu.physicalDevice, &gpu.physicalDeviceMemoryProperties);
		glslang_initialize_process();

		config.device = &gpu.device;
		config.queue = &gpu.queue;
		config.fence = &gpu.fence;
		config.commandPool = &gpu.commandPool;
		config.physicalDevice = &gpu.physicalDevice;
		config.isCompilerInitialized = 1;

		return res;
	}

	inline VkResult CreateBuffer(VkGPU &gpu, VkBuffer &buffer, uint64_t bufferSize) {
		VkResult res = VK_SUCCESS;

		VkBufferCreateInfo bufferInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = bufferSize,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};
		res = vkCreateBuffer(gpu.device, &bufferInfo, nullptr, &buffer);
		if (res) {
			std::print("Buffer creation failed, error code {}", static_cast<int>(res));
			return res;
		}
		
		VkCommandBufferAllocateInfo cmdAllocInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = gpu.commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		return res;
	}
	
	inline VkResult CreateCommandBuffer(VkGPU &gpu, VkCommandBuffer &commandBuffer) {
		VkResult res = VK_SUCCESS;

		VkCommandBufferAllocateInfo cmdAllocInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = gpu.commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		res = vkAllocateCommandBuffers(gpu.device, &cmdAllocInfo, &commandBuffer);
		if (res) {
			std::print("Command buffer allocation failed, error code {}", static_cast<int>(res));
			return res;
		}

		VkCommandBufferBeginInfo beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
		};
		res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		if (res) {
			std::print("Command buffer begin failed, error code {}", static_cast<int>(res));
			return res;
		}

		return res;
	}

	inline VkResult SetMemoryAllocInfo(VkMemoryPropertyFlags properties, const VkGPU &gpu, const VkBuffer &buffer, VkMemoryAllocateInfo &memAllocInfo) {
		VkResult res = VK_SUCCESS;
		bool found = false;

		VkBufferMemoryRequirementsInfo2 bufMemReqInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
			.buffer = buffer
		};
		VkMemoryRequirements2 memRequirements = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2
		};
		vkGetBufferMemoryRequirements2(gpu.device, &bufMemReqInfo, &memRequirements);
		VkPhysicalDeviceMemoryProperties2 memoryProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
		};
		vkGetPhysicalDeviceMemoryProperties2(gpu.physicalDevice, &memoryProperties);

		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.memoryRequirements.size;
		for (uint32_t i = 0; i < memoryProperties.memoryProperties.memoryTypeCount; i++) {
			if (((memRequirements.memoryRequirements.memoryTypeBits & (1 << i)) && (memoryProperties.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
				memAllocInfo.memoryTypeIndex = i;
				found = true;
				break;
			}
		}
		if (!found) {
			std::print("No suitable memory type found");
			return VK_ERROR_UNKNOWN;
		}

		return res;
	}

	inline VkResult AllocateMemory(const VkGPU &gpu, const VkBuffer &buffer, VkMemoryAllocateInfo &memAllocInfo, VkDeviceMemory &memory) {		
		VkResult res = VK_SUCCESS;
		res = vkAllocateMemory(gpu.device, &memAllocInfo, nullptr, &memory);
		if (res) {
			std::print("Buffer memory allocation failed, error code {}", static_cast<int>(res));
			return res;
		}

		VkBindBufferMemoryInfo bindInfo = {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.buffer = buffer,
			.memory = memory
		};
		vkBindBufferMemory2(gpu.device, 1, &bindInfo);

		return res;
	}
}
