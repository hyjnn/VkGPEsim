#pragma once

#include <stdfloat>
#include <concepts>

#include "MDVector.hpp"

namespace gpe {
// This is here to make clangd not complain about std::float32_t not existing
#ifdef __STDCPP_FLOAT32_T__
	using float32 = std::float32_t;
#else
	using float32 = float;
#endif

	template<std::floating_point float_type, unsigned dim>
	class Simulation {
	public:
	private:
		MDVector<float_type, dim> wavefunction;
		float_type dx, dt;
		void kick(float_type timestep); // kick (with imaginary time)
	};

	template<std::floating_point float_type, unsigned dim>
	inline void Simulation<float_type, dim>::kick(float_type timestep) {
		
	}
}
