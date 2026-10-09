#pragma once

#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <string>
#include <vulkan/vulkan_core.h>
#include <utils_vkFFT/utils_VkFFT.h>
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "MDVector.hpp"

// Units: hbar = m = a_s = 1 (this sets mass, length and time units)

/* Improvements and possible optimizations:
 * - Add multiple dimensions
 * - Add custom float types
 * - Overall code cleanup
 * - Use GPU (convolution) for kick pointwise mult
 * - Use GPU/CPU multithread for drift pointwise mult 
 * - Make potential a template argument or something (std::function is slower)
 */

namespace gpe {
	// Helper for constructing MDVectors that hold Wavefunctions

	inline void Normalize(MDVector<float> &wavefunc) {
		float norm = std::accumulate(wavefunc.begin(), wavefunc.end(), 0.f,
					     [](float acc, float next){ return acc + next*next; });
		norm = std::sqrt(norm);

		wavefunc /= norm;
	}

	// template<typename T>
	// InnerProduct(MDVector<T> phi, MDVector<T> psi) {
	// 	for (std::size_t i
	// }

	template<typename T>
	inline MDVector<T> CreateWavefuncVector(const std::vector<std::size_t> &sim_size) {
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

		float dx = 1, dt = 1, g = 4 * std::numbers::pi;
		std::size_t mid_index; // index corresponding to x=0, initialized as shape[i]/2
		std::function<float(float)> potential;

		VkResult WriteWavefunc(const MDVector<float> &source);
		VkResult ReadWavefunc(MDVector<float> &dest);
		void *MapMemory();
		void UnmapMemory();

		VkResult Kick(float time_step);
		void Drift(float time_step);

		void Run(std::size_t step_count);

	private:
		std::vector<std::uint64_t> shape;

		VkGPU gpu = {};
		VkFFTConfiguration config = {};
		VkFFTApplication app = {};
		VkFFTLaunchParams launchParams = {};
		VkMemoryAllocateInfo memAllocInfo = {};
		VkSubmitInfo submitInfo = {};
		VkBuffer buffer, kernelBuffer;
		VkCommandBuffer commandBuffer;
		VkDeviceMemory memory;

		// Translates grid index to physical position
		float PhysPos(std::size_t i);
		// Start simulation
		VkResult Start();
		// Wait for simulation to finish
		VkResult AwaitEnd();
	};

}
