#include <cstring>
#include <print>

#include <vector>
#include <vulkan/vulkan_core.h>

#include "sim.hpp"
#include "MDVector.hpp"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h"
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

namespace gpe {
	Simulation::Simulation(const std::vector<uint64_t> &dims) {
		config.FFTdim = dims.size();
		config.bufferSize = new uint64_t(sizeof(float) * 2);
		for (std::size_t i = 0; i < dims.size(); i++) {
			config.size[i] = dims[i];
			*config.bufferSize *= dims[i];
		}
		
		InitGPU();
		CreateBuffers();
		SetMemoryAllocInfo(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		AllocateMemory();
		initializeVkFFT(&app, config);
		
		launchParams.buffer = &buffer;
		launchParams.commandBuffer = &commandBuffer;
	}

	Simulation::~Simulation() {
		vkDestroyBuffer(gpu.device, buffer, nullptr);
		vkFreeMemory(gpu.device, memory, nullptr);
		deleteVkFFT(&app);
	}

	VkResult Simulation::WriteWavefunc(const MDVector<float>& source) {
		VkResult res = VK_SUCCESS;
		void *raw_data;

		res = vkMapMemory(gpu.device, memory, 0, *config.bufferSize, 0, &raw_data);
		if (res) {
			std::print("Mapping memory failed, error code {}\n", static_cast<int>(res));
			return res;
		}
		std::memcpy(raw_data, source.data.data(), *config.bufferSize);
		vkUnmapMemory(gpu.device, memory);

		return res;
	}

	VkResult Simulation::ReadWavefunc(MDVector<float>& dst) {
		VkResult res = VK_SUCCESS;
		void *raw_data;
	
		res = vkMapMemory(gpu.device, memory, 0, *config.bufferSize, 0, &raw_data);
		if (res) {
			std::print("Mapping memory failed, error code {}\n", static_cast<int>(res));
			return res;
		}
		std::memcpy(dst.data.data(), raw_data, *config.bufferSize);
		vkUnmapMemory(gpu.device, memory);

		return res;
	}

	VkResult Simulation::QueueKick(float dt) {
		VkFFTResult res = VKFFT_SUCCESS;

		res = VkFFTAppend(&app, -1, &launchParams);
		if (res) {
			std::print("VkFFTAppend failed, error code {}\n", static_cast<int>(res));
			return VK_ERROR_UNKNOWN;
		}

		return VK_SUCCESS;
	}

	VkResult Simulation::Start() {
		VkResult res = VK_SUCCESS;
		
		res = vkEndCommandBuffer(commandBuffer);
		if (res) {
			std::print("Command buffer end failed, error code {}\n", static_cast<int>(res));
			return res;
		}

		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		res = vkQueueSubmit(gpu.queue, 1, &submitInfo, gpu.fence);
		if (res) {
			std::print("Queue submission failed, error code {}\n", static_cast<int>(res));
			return res;
		}

		return res;
	}

	VkResult Simulation::AwaitEnd() {
		VkResult res = VK_SUCCESS;

		res = vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, UINT64_MAX);
		if (res) {
			std::print("Wait for fences failed, error code {}\n", static_cast<int>(res));
			return res;
		}

		return res;
	}

	VkResult Simulation::InitGPU() {
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

	VkResult Simulation::CreateBuffers() {
		VkResult res = VK_SUCCESS;

		VkBufferCreateInfo bufferInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = *config.bufferSize,
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

	VkResult Simulation::SetMemoryAllocInfo(VkMemoryPropertyFlags properties) {
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

	VkResult Simulation::AllocateMemory() {		
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
