#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <format>
#include <fstream>
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
	namespace {
		// Saves shape vector to numpy .npy format friendly string 
		std::string ShapeToNpy(std::vector<std::size_t> shape) {
			if (shape.size() == 0) return "(0,)";
			if (shape.size() == 1) return std::format("({},)", shape[0] / 2);
			std::string output = std::format("({:d}", shape[0] / 2);

			for (auto i = shape.begin() + 1; i != shape.end(); i++) {
				output += std::format(", {:d}", *i);
			}
			output += ")";

			return output;
		}
	}

	bool SaveWavefunc(const MDVector<float> &src, std::string filename, bool overwrite) {
		std::ios::openmode mode = std::ios::binary;
		mode |= (overwrite ? std::ios::trunc : std::ios::noreplace);
		std::ofstream outfile(filename, mode);

		if (!outfile.is_open()) {
			std::print("Failed to open file {}", filename);
			return false;
		}
		
		std::string shape = ShapeToNpy(src.shape());
		// For now assumes we are using 32-bit floats
		std::string header = "{'descr': '<c8',"
				     " 'fortran_order': True,"
				     " 'shape': " + shape + ", }";

		unsigned pad_len = (header.size() + 10) % 64;
		pad_len = 63 - pad_len;
		header += std::string(pad_len, '\x20'); // Pad with spaces
		header += '\n';
		
		std::array<char, 2> header_len_bytes = { static_cast<char>(header.size() & 0xff),
		                                         static_cast<char>((header.size() >> 8) & 0xff) };
		std::string preheader = std::string("\x93NUMPY\x01") + '\x00' + header_len_bytes[0] + header_len_bytes[1];
		outfile << preheader + header;	
		outfile.write(reinterpret_cast<const char *>(src.raw_data()), src.size() * sizeof(float));

		return true;
	}

	Simulation::Simulation(const std::vector<uint64_t> &dims) {
		config.FFTdim = dims.size();
		shape = dims;
		mid_index = dims[0] / 2;
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

		for (std::size_t i = 0; i < N; i += 2) {
			float k = 2 * std::numbers::pi * i / N / dx;
			if (i > N / 2) {
				k -= std::numbers::pi / dx;
			}
			std::complex<float> ekin = -1.0f * k*k * time_step / 2;
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

	void Simulation::Drift(float time_step) {
		using namespace std::complex_literals;

		void *raw_data = MapMemory();
		float *float_data = static_cast<float*>(raw_data);

		std::size_t N = *config.bufferSize / sizeof(float);
		
		float norm = 0;
		for (std::size_t i = 0; i < N; i += 2) {
			std::complex<float> epot = -1.0f * potential(PhysPos(i / 2)) * time_step;
			std::complex<float> eigenexp = std::exp(epot);
			std::complex<float> prod = (float_data[i] + float_data[i+1]*1.0if) * eigenexp;
			float_data[i] = prod.real();
			float_data[i+1] = prod.imag();

			norm += float_data[i]*float_data[i] + float_data[i+1]*float_data[i+1];
		}
		norm = std::sqrt(norm*dx);
		
		// For numerical stability
		std::transform(float_data, float_data + N, float_data, [norm](float val){ return val / norm; });

		UnmapMemory();
	}

	void Simulation::Run(std::size_t step_count) {
		for (std::size_t step = 0; step < step_count; step++) {
			Drift(dt / 2);
			Kick(dt);
			Drift(dt / 2);
		}
	}

	float Simulation::PhysPos(std::size_t i) {
		return i * dx - mid_index * dx;
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
