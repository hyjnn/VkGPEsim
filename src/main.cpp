#include "sim.hpp"
#include "MDVector.hpp"
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <numeric>
#include <print>
#include <random>

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

void HarmonicBaseState(MDVector<float> &wavefunc, float omega, float dx) {
	float A = std::pow(omega / std::numbers::pi, .25f);
	float exponent = -1 * omega / 2;
	
	std::size_t N = wavefunc.size() / 2;
	for (std::size_t i = 0; i < wavefunc.size(); i += 2) {
		float x = i * dx/2 - N * dx / 2;
		float exp = std::exp(exponent * x*x);
		wavefunc[i] = A*exp;
	}
}

void RandomState(MDVector<float> &wavefunc) {
	std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

	for (auto &val : wavefunc) {
		val = rng() / static_cast<float>(rng.max());
	}

	gpe::Normalize(wavefunc);
}

void ProjectPerp(MDVector<float> &projectee, MDVector<float> projector) {
	std::complex<float> prod = gpe::InnerProduct(projectee, projector);
	gpe::Scale(projector, prod);
	projectee -= projector;
}

struct NullPotential {
	float operator()(float x) {
		return 0.f;
	}
};

struct HarmonicPotential {
	float coeff = 0.5f; // In our units this is just omega^2/2, base-state energy is sqrt(coeff/2)
	float operator()(float x) {
		return coeff*x*x;
	}
};

struct ModifiedHarmonicPotential {
	float coeff = 0.5f;
	float gauge = 0.5f;
	float operator()(float x) {
		return coeff*x*x - gauge;
	}
};

int main() {
	float space_length, space_step_ratio, g;
	std::cout << "Space length:\n>";
	std::cin >> space_length;
	std::cout << "g:\n>";
	std::cin >> g;
	std::cout << "Space step ratio:\n>";
	std::cin >> space_step_ratio;
	gpe::Simulation testsim = gpe::CreateGPESim(space_length, g, space_step_ratio);


	std::size_t sim_length;
	std::cout << "Sim length:\n>";
	std::cin >> sim_length;

	auto potential = ModifiedHarmonicPotential();
	std::cout << "Potential coeff:\n>";
	std::cin >> potential.coeff;
	potential.gauge = 3*std::sqrt(potential.coeff/2);
	testsim.potential = NullPotential();
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(testsim.shape());

	RandomState(wavefunc);
	
	testsim.WriteWavefunc(wavefunc);
	testsim.Run(sim_length);
	testsim.ReadWavefunc(wavefunc);

	std::print("Base state norm: {}\n", std::accumulate(wavefunc.begin(), wavefunc.end(), 0.f, 
	           [](float acc, float next){ return acc + next*next; })*testsim.dx);
	
	gpe::SaveWavefunc(wavefunc, "results/test.npy");
}
