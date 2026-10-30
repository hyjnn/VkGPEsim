#include "sim.hpp"
#include "MDVector.hpp"
#include "config.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <string>

using namespace std::complex_literals;

void MomentumState(MDVector<float> &wavefunc, unsigned mode, float dx) {
	std::complex<float> exponent = 2if * (std::numbers::pi_v<float> * mode);
	for (std::size_t i = 0; i < wavefunc.size(); i += 2) {
		std::complex<float> exp = std::exp(exponent * (static_cast<float>(i) / wavefunc.size()));
		wavefunc[i] = exp.real();
		wavefunc[i+1] = exp.imag();
	}

	gpe::Normalize(wavefunc, dx);
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

void RandomState(MDVector<float> &wavefunc, float dx) {
	std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

	for (auto &val : wavefunc) {
		val = rng() / static_cast<float>(rng.max());
	}

	gpe::Normalize(wavefunc, dx);
}

void DoubleSoliton(MDVector<float> &wavefunc, float distance, float healing_length, float dx) {
	healing_length /= dx;
	float centre = wavefunc.size() / 2. - distance / dx;

	std::size_t i = 0;

	for (; i < wavefunc.size() / 2; i += 2) {
		wavefunc[i] = std::tanh((i - centre) / healing_length);
	}

	for (; i < wavefunc.size(); i += 2) {
		wavefunc[i] = std::tanh((-i + wavefunc.size() - centre) / healing_length);
	}

	gpe::Normalize(wavefunc, dx);
}

void KickState(MDVector<float> &wavefunc, float dp, float dx) {
	for (size_t i = 0; i < wavefunc.size(); i += 2) {
		std::complex<float> prod = wavefunc[i] + wavefunc[i+1] * 1.0if;
		prod *= std::exp(1.0if * dp * (i*dx/2));
		wavefunc[i] = prod.real();
		wavefunc[i+1] = prod.imag();
	}
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

struct GaugedHarmonicPotential {
	float coeff = 0.5f;
	float gauge = 0.5f;
	float operator()(float x) {
		return coeff*x*x - gauge;
	}
};

int main(int argc, char **argv) {
	float space_length, space_step_ratio, g, time_step, soliton_distance;
	std::size_t sim_length;
	std::string config_filename = gpe::GetConfigFile(argc, argv);

	if (config_filename.at(0) == '\0') {
		std::print("{}", config_filename.substr(1));
		return 0;
	}

	gpe::ParamReader parser;
	parser.DefineParams("x_span", space_length, "g", g,
	                    "x_step_ratio", space_step_ratio, "sim_duration", sim_length,
			    "time_step", time_step, "soliton_distance", soliton_distance);

	parser.ReadParams(std::ifstream(config_filename));

	gpe::Simulation testsim = gpe::CreateGPESim(space_length, g, space_step_ratio, time_step);
	testsim.dt *= -1.0if;
	testsim.potential = NullPotential();
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(testsim.shape());

	DoubleSoliton(wavefunc, soliton_distance, std::sqrt(space_length / g), testsim.dx);
	// RandomState(wavefunc, testsim.dx);
	// HarmonicBaseState(wavefunc, 1, testsim.dx);
	// KickState(wavefunc, 1, testsim.dx);

	testsim.WriteWavefunc(wavefunc);
	testsim.Run(5*sim_length);
	testsim.dt *= 1.0if;
	testsim.Run(sim_length);
	testsim.ReadWavefunc(wavefunc);

	std::print("Base state norm: {}\n", std::accumulate(wavefunc.begin(), wavefunc.end(), 0.f, 
	           [](float acc, float next){ return acc + next*next; })*testsim.dx);
	
	gpe::SaveWavefunc(wavefunc, "results/test.npy");
}
