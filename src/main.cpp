#include "sim.hpp"
#include "MDVector.hpp"
#include <complex>
#include <numbers>
#include <numeric>
#include <print>
#include <vector>

void MomentumState(MDVector<float> &wavefunc, unsigned mode) {
	using namespace std::complex_literals;

	std::complex<float> exponent = 2if * (std::numbers::pi_v<float> * mode);
	for (std::size_t i = 0; i < wavefunc.data.size(); i += 2) {
		std::complex<float> exp = std::exp(exponent * (static_cast<float>(i) / wavefunc.data.size()));
		wavefunc[i] = exp.real();
		wavefunc[i+1] = exp.imag();
	}
}

int main() {
	std::vector<std::size_t> sim_size = { 50 };
	gpe::Simulation testsim(sim_size);
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(sim_size);
	
	MomentumState(wavefunc, 2);
	for (std::size_t i = 0; i < wavefunc.shape[0]; i++) {
		std::print("{} ", wavefunc[i]);
	}
	std::print("\n");

	testsim.WriteWavefunc(wavefunc);
	testsim.Kick(sim_size[0] * sim_size[0] / std::numbers::pi);
	testsim.ReadWavefunc(wavefunc);

	for (std::size_t i = 0; i < wavefunc.shape[0]; i++) {
		std::print("{} ", wavefunc[i]);
	}
	std::print("\n");
	std::print("Norm: {}\n", std::accumulate(wavefunc.data.begin(), wavefunc.data.end(), 0.f, 
	           [](float acc, float next){ return acc + next*next; }));

}
