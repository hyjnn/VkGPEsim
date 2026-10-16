#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <vulkan/vulkan_core.h>
#include <utils_vkFFT/utils_VkFFT.h>
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "MDVector.hpp"

// Units: hbar = m = a_s = 1 (this sets mass, length and time units)

/* Improvements and possible optimizations:
 * - Add multiple dimensions
 * - Add mid step
 * - Add custom float types
 * - Overall code cleanup
 * - Use GPU (convolution) for kick pointwise mult
 * - Use GPU/CPU multithread for drift pointwise mult 
 * - Make potential a template argument or something (std::function is slower)
 * - I use std::complex in some calculations. This almost always involves unnecessary copies.
 */

namespace gpe {

	inline void Normalize(MDVector<float> &wavefunc) {
		float norm = std::accumulate(wavefunc.begin(), wavefunc.end(), 0.f,
					     [](float acc, float next){ return acc + next*next; });
		norm = std::sqrt(norm);

		wavefunc /= norm;
	}

	template<typename T>
	std::complex<float> InnerProduct(MDVector<T> phi, MDVector<T> psi) {
		using namespace std::complex_literals;

		std::complex<float> result = 0;
		for (std::size_t i = 0; i < phi.size(); i += 2) {
			std::complex<float> phi_val = phi[i] + phi[i+1] * 1.fi;
			std::complex<float> psi_val = psi[i] + psi[i+1] * 1.fi;

			result += phi_val * psi_val;
		}

		return result;
	}

	template<typename T>
	void Scale(MDVector<T> phi, std::complex<T> scalar) {
		using namespace std::complex_literals;

		for (std::size_t i = 0; i < phi.size(); i += 2) {
			phi[i] = phi[i] * scalar.real() - phi[i+1] * scalar.imag();
			phi[i+1] = phi[i] * scalar.imag() + phi[i+1] * scalar.real();
		}
	}

	// Helper for constructing MDVectors that hold Wavefunctions
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

		float dx = 1, dt = 1, g = 4 * std::numbers::pi * 2;
		std::size_t mid_index; // index corresponding to x=0, initialized as shape[i]/2
		std::function<float(float)> potential;

		const std::vector<std::size_t> &shape() {
			return m_shape;
		}

		VkResult WriteWavefunc(const MDVector<float> &source);
		VkResult ReadWavefunc(MDVector<float> &dest);
		void *MapMemory();
		void UnmapMemory();

		VkResult Kick(float time_step);
		void Drift(float time_step);

		void Run(std::size_t step_count);

	private:
		std::vector<std::uint64_t> m_shape;

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
	
	// Create GP eq simulation with dx = xi/r
	inline Simulation CreateGPESim(float length, float g, float r, std::optional<float> dt = std::nullopt) {
		float dx = std::sqrt(.5 * length / g) / r;

		Simulation result({ static_cast<std::size_t>(length / dx) });
		result.dx = dx;
		result.g = g;

		if (!dt) dt = 2 * dx*dx / 5;
		result.dt = *dt;

		return result;
	}
}
