#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <format>
#include <functional>
#include <numbers>
#include <numeric>
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
		m_shape = dims;
		zero_index.resize(dims.size());
		m_dx.resize(dims.size());
		m_dp.resize(dims.size());
		config.bufferSize = new uint64_t(sizeof(float) * 2);
		for (std::size_t i = 0; i < dims.size(); i++) {
			zero_index[i] = dims[i] / 2;
			m_dx[i] = 1;
			m_dp[i] = 2*std::numbers::pi/m_dx[i]/m_shape[i];
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
#ifndef NDEBUG
		std::print("Using GPU ID: {}\n", gpu.physicalDeviceProperties.deviceName);
#endif
		launchParams.buffer = &buffer;
		launchParams.commandBuffer = &commandBuffer;
	}

	Simulation::~Simulation() {
		vkDestroyBuffer(gpu.device, buffer, nullptr);
		vkFreeMemory(gpu.device, memory, nullptr);
		deleteVkFFT(&app);
	}

	void Simulation::Setdx(std::vector<float> new_dx) {
		m_dx = new_dx;
		std::ranges::transform(m_dx, m_shape, m_dp.begin(), [](float dx, float N)
		                                                    { return 2 * std::numbers::pi / dx / N; });
	}

	const std::vector<float> &Simulation::Getdx() const {
		return m_dx;
	}

	const float &Simulation::Getdx(std::size_t i) const {
		return m_dx.at(i);
	}

	float Simulation::GetdV() const {
		return std::reduce(m_dx.begin(), m_dx.end(), 1.0f, std::multiplies<float>()); 
	}

	VkResult Simulation::WriteWavefunc(const MDVector<float>& source) {
		VkResult res = VK_SUCCESS;
		void *raw_data;

		res = vkMapMemory(gpu.device, memory, 0, *config.bufferSize, 0, &raw_data);
		if (res) {
			std::print("Mapping memory failed, error code {}\n", static_cast<int>(res));
			return res;
		}
		std::memcpy(raw_data, source.raw_data(), *config.bufferSize);
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
		std::memcpy(dst.raw_data(), raw_data, *config.bufferSize);
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

	VkResult Simulation::Kick(std::complex<float> time_step) {
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

		for (std::size_t i = 0; i < N; i += 2) {
			float k_sqr = MomentumNormSqr(i / 2);
			std::complex<float> ekin = -1.0if * k_sqr * time_step / 2.f;
			std::complex<float> eigenexp = std::exp(ekin);
			std::complex<float> prod = (float_data[i] + float_data[i+1]*1.0if) * eigenexp;
			float_data[i] = prod.real();
			float_data[i+1] = prod.imag();
		}

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

	void Simulation::Drift(std::complex<float> time_step) {
		using namespace std::complex_literals;

		void *raw_data = MapMemory();
		float *float_data = static_cast<float*>(raw_data);

		std::size_t N = *config.bufferSize / sizeof(float);
		
		float norm = 0;
		for (std::size_t i = 0; i < N; i += 2) {
			std::complex<float> epot = potential(PhysPos(i / 2));
			float density = float_data[i]*float_data[i] + float_data[i+1]*float_data[i+1];
			epot += g * density;
			std::complex<float> eigenexp = std::exp(-1.if * epot * time_step);
			std::complex<float> prod = (float_data[i] + float_data[i+1]*1.0if) * eigenexp;
			float_data[i] = prod.real();
			float_data[i+1] = prod.imag();

			if (time_step.imag()) {
				norm += float_data[i]*float_data[i] + float_data[i+1]*float_data[i+1];
			}
		}
		// For numerical stability
		if (time_step.imag()) {
			norm = std::sqrt(norm*GetdV());
			std::transform(float_data, float_data + N, float_data,
				       [norm](float val){ return val / norm; });
		}
		UnmapMemory();
	}

	void Simulation::Run(std::size_t step_count) {
		for (std::size_t step = 0; step < step_count; step++) {
			Drift(dt / 2.f);
			Kick(dt);
			Drift(dt / 2.f);
		}
	}

	std::vector<float> Simulation::PhysPos(std::size_t i) const {
		std::vector<float> pos(m_shape.size(), 0);

		for (std::size_t j = pos.size(); j--;) {
			pos[j] += (i % m_shape[j]) * m_dx[j];
			pos[j] -= zero_index[j] * m_dx[j];
			i /= m_shape[j];
		}

		return pos;
	}

	float Simulation::MomentumNormSqr(std::size_t i) const {			
		float k2 = 0;

		for (std::size_t j = 0; j < m_dx.size(); j++) {
			long eff_i = i % m_shape[j];
			if (eff_i > m_shape[j]/2) {
				eff_i -= m_shape[j]; // Get negative momenta
			}

			float k_comp = eff_i * m_dp[j];
			k2 += k_comp*k_comp;
			i /= m_shape[j]; // Get ready for next dimension
		}

		return k2;
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
			std::print("Command buffer begin failed, error code {}\n", static_cast<int>(res));
			return res;
		}

		return res;
	}
}
