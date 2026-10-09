#include "sim.hpp"
#include "MDVector.hpp"
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <vector>

void MomentumState(MDVector<float> &wavefunc, unsigned mode) {
	using namespace std::complex_literals;

	std::complex<float> exponent = 2if * (std::numbers::pi_v<float> * mode);
	for (std::size_t i = 0; i < wavefunc.size(); i += 2) {
		std::complex<float> exp = std::exp(exponent * (static_cast<float>(i) / wavefunc.size()));
		wavefunc[i] = exp.real();
		wavefunc[i+1] = exp.imag();
	}

	gpe::Normalize(wavefunc);
}

void HarmonicBaseState(MDVector<float> &wavefunc, float omega) {
	float A = std::pow(omega / std::numbers::pi, .25f);
	std::complex<float> exponent = -1 * omega / 2;
	
	for (std::size_t i = 0; i < wavefunc.size(); i += 2) {
		float x = static_cast<float>(i) / wavefunc.size();
		std::complex<float> exp = std::exp(exponent * x*x);
		wavefunc[i] = A*exp.real();
		wavefunc[i+1] = A*exp.imag();
	}
}

void RandomState(MDVector<float> &wavefunc) {
	std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

	for (auto &val : wavefunc) {
		val = rng() / static_cast<float>(rng.max());
	}

	gpe::Normalize(wavefunc);
}

struct HarmonicPotential {
	float coeff = 0.5f; // In our units this is just omega^2/2, base-state energy is sqrt(coeff/2)
	float operator()(float x) {
		return coeff*x*x;
	}
};

int main() {
	std::vector<std::size_t> sim_size = { 50 };
	gpe::Simulation testsim(sim_size);
	std::cout << "Time step:\n>";
	std::cin >> testsim.dt;
	std::size_t sim_length;
	std::cout << "Sim length:\n>";
	std::cin >> sim_length;

	auto potential = HarmonicPotential();
	std::cout << "Potential coeff:\n>";
	std::cin >> potential.coeff;
	testsim.potential = potential;
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(sim_size);
	MDVector<float> base_wavefunc = gpe::CreateWavefuncVector<float>(sim_size);

	RandomState(wavefunc);
	HarmonicBaseState(base_wavefunc, potential.coeff);
	
	
	testsim.WriteWavefunc(wavefunc);
	testsim.Run(sim_length);
	testsim.ReadWavefunc(wavefunc);

	std::print("Norm: {}\n", std::accumulate(wavefunc.begin(), wavefunc.end(), 0.f, 
	           [](float acc, float next){ return acc + next*next; }));
	
	gpe::SaveWavefunc(wavefunc, "results/test.npy");
}
