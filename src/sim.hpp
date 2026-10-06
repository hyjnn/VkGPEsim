#pragma once

#include <cstdint>
#include <numeric>
#include <string>
#include <vulkan/vulkan_core.h>
#include <utils_vkFFT/utils_VkFFT.h>
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "MDVector.hpp"

namespace gpe {
	// Helper for constructing MDVectors that hold Wavefunctions

	inline void Normalize(MDVector<float> &wavefunc) {
		float norm = std::accumulate(wavefunc.data.begin(), wavefunc.data.end(), 0.f,
					     [](float acc, float next){ return acc + next*next; });
		norm = std::sqrt(norm);

		wavefunc /= norm;
	}

	template<typename T>
	MDVector<T> CreateWavefuncVector(const std::vector<std::size_t> &sim_size) {
		std::vector<std::size_t> real_size(sim_size.size());
		for (std::size_t i = 0; i < sim_size.size(); i++) {
			real_size[i] = 2 * sim_size[i];
		}

		return MDVector<T>(real_size, 0);
	}

	bool SaveWavefunc(const MDVector<float> &src, std::string filename, bool overwrite = true);

	class Simulation {
	public:
		Simulation(const std::vector<std::uint64_t> &dims);
		~Simulation();

		VkResult WriteWavefunc(const MDVector<float> &source);
		VkResult ReadWavefunc(MDVector<float> &dest);
		void *MapMemory();
		void UnmapMemory();

		VkResult Kick(float dt);
	private:
		std::vector<std::uint64_t> shape;
		float dx = 1, dt = 1;
		
		VkGPU gpu = {};
		VkFFTConfiguration config = {};
		VkFFTApplication app = {};
		VkFFTLaunchParams launchParams = {};
		VkMemoryAllocateInfo memAllocInfo = {};
		VkSubmitInfo submitInfo = {};
		VkBuffer buffer, kernelBuffer;
		VkCommandBuffer commandBuffer;
		VkDeviceMemory memory;

		// Start simulation
		VkResult Start();
		// Wait for simulation to finish
		VkResult AwaitEnd();
	};

}
