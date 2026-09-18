#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vkFFT/vkFFT.h>
#include <print>

#include <vulkan/vulkan_core.h>
#include "utils_vkFFT/utils_VkFFT.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h"
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

int main() {
	VkResult res = VK_SUCCESS;
	VkGPU gpu = {};

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

	VkFFTConfiguration config = {};
	VkFFTApplication app = {};
	
	config.FFTdim = 1;
	config.size[0] = 100;
	uint64_t bufferSize = static_cast<uint64_t>(sizeof(float)) * 2 * config.size[0];

	config.device = &gpu.device;
	config.queue = &gpu.queue;
	config.fence = &gpu.fence;
	config.commandPool = &gpu.commandPool;
	config.physicalDevice = &gpu.physicalDevice;
	config.isCompilerInitialized = 1;
	
	// Buffer creation
	
	VkBuffer buffer;
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = bufferSize;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	res = vkCreateBuffer(gpu.device, &bufferInfo, nullptr, &buffer);
	if (res) {
		std::print("Buffer creation failed, error code {}", static_cast<int>(res));
	}

	// Buffer allocation
	
	VkBufferMemoryRequirementsInfo2 bufMemReqInfo = {};
	bufMemReqInfo.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
	bufMemReqInfo.buffer = buffer;
	VkMemoryRequirements2 memoryRequirements = {};
	memoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
	vkGetBufferMemoryRequirements2(gpu.device, &bufMemReqInfo, &memoryRequirements);
	VkPhysicalDeviceMemoryProperties2 memoryProperties = {};
	memoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
	vkGetPhysicalDeviceMemoryProperties2(gpu.physicalDevice, &memoryProperties);
	VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VkMemoryAllocateInfo memAllocInfo = {};
	memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllocInfo.allocationSize = memoryRequirements.memoryRequirements.size;
	bool found = false;
	for (uint32_t i = 0; i < memoryProperties.memoryProperties.memoryTypeCount; i++) {
		if (((memoryRequirements.memoryRequirements.memoryTypeBits & (1 << i)) && (memoryProperties.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
			memAllocInfo.memoryTypeIndex = i;
			found = true;
			break;
		}
	}
	if (!found) {
		std::print("No suitable memory type found");
		return -1;
	}
	VkDeviceMemory memory;
	res = vkAllocateMemory(gpu.device, &memAllocInfo, nullptr, &memory);
	if (res) {
		std::print("Buffer memory allocation failed, error code {}", static_cast<int>(res));
	}
	VkBindBufferMemoryInfo bindInfo = {};
	bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
	bindInfo.buffer = buffer;
	bindInfo.memory = memory;
	vkBindBufferMemory2(gpu.device, 1, &bindInfo);

	// Write to buffer

	VkMemoryMapInfo memMapInfo = {};
	memMapInfo.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
	memMapInfo.memory = memory;
	memMapInfo.size = bufferInfo.size;
	void *data;
	res = vkMapMemory(
	    gpu.device,
	    memory,
	    0,
	    bufferInfo.size,
	    0,
	    &data
	);
	*static_cast<float*>(data) = 1.;
	VkMemoryUnmapInfo memUnmapInfo = {};
	memUnmapInfo.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO;
	memUnmapInfo.memory = memory;
	vkUnmapMemory2(gpu.device, &memUnmapInfo);

	// Command buffer allocation

	VkCommandBuffer commandBuffer;
	VkCommandBufferAllocateInfo cmdAllocInfo = {};
	cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdAllocInfo.commandPool = gpu.commandPool;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = 1;
	res = vkAllocateCommandBuffers(gpu.device, &cmdAllocInfo, &commandBuffer);
	if (res) {
		std::print("Command buffer allocation failed, error code {}", static_cast<int>(res));
	}

	config.bufferSize = &bufferSize;

	VkFFTResult resFFT = initializeVkFFT(&app, config);
	VkFFTLaunchParams launchParams = {};
	launchParams.buffer = &buffer;
	launchParams.commandBuffer = &commandBuffer;
	resFFT = VkFFTAppend(&app, -1, &launchParams);

	// Reading buffer???
	
	vkMapMemory2(gpu.device, &memMapInfo, &data);
	float fdata[bufferInfo.size / sizeof(float)];
	std::memcpy(static_cast<void*>(fdata), data, bufferInfo.size);
	vkUnmapMemory2(gpu.device, &memUnmapInfo);
	
	for (unsigned i = 0; i < bufferInfo.size / sizeof(float); i++) {
		std::print("{}", fdata[i]);
	}

	// Clearing buffer
	
	vkDestroyBuffer(gpu.device, buffer, nullptr);
	vkFreeMemory(gpu.device, memory, nullptr);

	deleteVkFFT(&app);
}
