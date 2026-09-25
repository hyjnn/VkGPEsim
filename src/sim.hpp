#pragma once

#include <vulkan/vulkan_core.h>
#include <utils_vkFFT/utils_VkFFT.h>
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "MDVector.hpp"

namespace gpe {
	class Simulation {
	public:
		Simulation(const std::vector<uint64_t> &dims);
		~Simulation();

		VkResult WriteWavefunc(const MDVector<float> &source);
		VkResult ReadWavefunc(MDVector<float> &dest);

		VkResult QueueKick(float dt);
		// Start simulation
		VkResult Start();
		// Wait for simulation to finish
		VkResult AwaitEnd();
	private:
		float dx, dt;
		
		
		VkGPU gpu = {};
		VkFFTConfiguration config = {};
		VkFFTApplication app = {};
		VkFFTLaunchParams launchParams = {};
		VkMemoryAllocateInfo memAllocInfo = {};
		VkSubmitInfo submitInfo = {};
		VkBuffer buffer;
		VkCommandBuffer commandBuffer;
		VkDeviceMemory memory;

		VkResult InitGPU();

		VkResult CreateBuffers();
		VkResult SetMemoryAllocInfo(VkMemoryPropertyFlags properties);
		VkResult AllocateMemory();
	};

}
