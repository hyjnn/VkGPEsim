#include <cstring>
#include <print>

#include <vector>
#include <vulkan/vulkan_core.h>
#include <vkFFT/vkFFT.h>

#include "sim.hpp"
#include "MDVector.hpp"
#include "vkHelpers.hpp"

namespace gpe {
	Simulation::Simulation(const std::vector<uint64_t> &dims) {
		config.FFTdim = dims.size();
		config.bufferSize = new uint64_t(sizeof(float) * 2);
		for (std::size_t i = 0; i < dims.size(); i++) {
			config.size[i] = dims[i];
			*config.bufferSize *= dims[i];
		}
		
		::gpe::InitGPU(gpu, config);
		CreateBuffer(gpu, buffer, *config.bufferSize);
		CreateCommandBuffer(gpu, commandBuffer);
		::gpe::SetMemoryAllocInfo(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		                          gpu, buffer, memAllocInfo);
		::gpe::AllocateMemory(gpu, buffer, memAllocInfo, memory);
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
}
