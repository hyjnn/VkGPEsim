#include <cmath>
#include <cstring>
#include <complex>
#include <numbers>
#include <print>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <vkFFT/vkFFT.h>

#include "sim.hpp"
#include "MDVector.hpp"
#include "vkHelpers.hpp"

namespace gpe {
	Simulation::Simulation(const std::vector<uint64_t> &dims) {
		config.FFTdim = dims.size();
		shape = dims;
		config.bufferSize = new uint64_t(sizeof(float) * 2);
		for (std::size_t i = 0; i < dims.size(); i++) {
			config.size[i] = dims[i];
			*config.bufferSize *= dims[i];
		}
		config.normalize = 1;
		
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

	void *Simulation::MapMemory() {
		VkResult res = VK_SUCCESS;
		void *raw_data;
	
		res = vkMapMemory(gpu.device, memory, 0, *config.bufferSize, 0, &raw_data);
		if (res) {
			std::print("Mapping memory failed, error code {}\n", static_cast<int>(res));
		}
		
		return raw_data;
	}

	void Simulation::UnmapMemory() {
		vkUnmapMemory(gpu.device, memory);
	}

	VkResult Simulation::Kick(float time_step) {
		using namespace std::complex_literals;

		VkFFTResult fftres = VKFFT_SUCCESS;
		VkResult res = VK_SUCCESS;

		fftres = VkFFTAppend(&app, -1, &launchParams);
		if (fftres) {
			std::print("VkFFTAppend failed, error code {}\n", static_cast<int>(fftres));
			return VK_ERROR_UNKNOWN;
		} // Unnormalized wavefunc now

		Start();
		AwaitEnd();
		
		void *raw_data = MapMemory();
		float *float_data = static_cast<float*>(raw_data);

		std::size_t N = *config.bufferSize / sizeof(float);

		std::string pre_mult = "", post_mult = "";

		for (std::size_t i = 0; i < N; i += 2) {
			pre_mult += std::format("{} {} ", float_data[i], float_data[i+1]);
			float k = 2 * std::numbers::pi * i / N / dx;
			std::complex<float> ekin = -1.0fi * k*k * time_step / 2;
			std::complex<float> eigenexp = std::exp(ekin);
			std::complex<float> prod = (float_data[i] + float_data[i+1]*1.0if) * eigenexp;
			float_data[i] = prod.real();
			float_data[i+1] = prod.imag();
			post_mult += std::format("{} {} ", float_data[i], float_data[i+1]);
		}
		
		// std::print("{}", pre_mult + '\n');
		// std::print("{}", post_mult + '\n');

		UnmapMemory();

		fftres = VkFFTAppend(&app, 1, &launchParams);
		if (fftres) {
			std::print("VkFFTAppend failed, error code {}\n", static_cast<int>(fftres));
			return VK_ERROR_UNKNOWN;
		} // Normalized wavefunc now
		
		Start();
		AwaitEnd();
		
		return res;
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
}
