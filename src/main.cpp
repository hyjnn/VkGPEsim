#include "sim.hpp"
#include "MDVector.hpp"
#include <chrono>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <vector>

void MomentumState(MDVector<float> &wavefunc, unsigned mode) {
	using namespace std::complex_literals;

	std::complex<float> exponent = 2if * (std::numbers::pi_v<float> * mode);
	for (std::size_t i = 0; i < wavefunc.data.size(); i += 2) {
		std::complex<float> exp = std::exp(exponent * (static_cast<float>(i) / wavefunc.data.size()));
		wavefunc[i] = exp.real();
		wavefunc[i+1] = exp.imag();
	}

	gpe::Normalize(wavefunc);
}

void RandomState(MDVector<float> &wavefunc) {
	std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

	for (auto &val : wavefunc.data) {
		val = rng() / static_cast<float>(rng.max());
	}

	gpe::Normalize(wavefunc);
}

int main() {
	std::vector<std::size_t> sim_size = { 50 };
	gpe::Simulation testsim(sim_size);
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(sim_size);

	RandomState(wavefunc);
	
	testsim.WriteWavefunc(wavefunc);
	for (std::size_t i = 0; i < 0; i++) {
		testsim.Kick(sim_size[0] * sim_size[0] / std::numbers::pi / 2.);
	}
	testsim.ReadWavefunc(wavefunc);

	std::print("Norm: {}\n", std::accumulate(wavefunc.data.begin(), wavefunc.data.end(), 0.f, 
	           [](float acc, float next){ return acc + next*next; }));
	
	gpe::SaveWavefunc(wavefunc, "results/test.npy");
}
