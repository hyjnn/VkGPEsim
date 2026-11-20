#include "sim.hpp"
#include "MDVector.hpp"
#include "config.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <string>

using namespace std::complex_literals;

float IntegerPower(float base, unsigned long exp) {
	if (exp == 0) return 1; // 0^0=1 i guess whatever
	if (exp == 1) return base;
	float result = base;

	for (unsigned long mask = 1ul << (std::bit_width(exp)-2); mask; mask >>= 1) {
		result *= result;
		if (exp & mask) result *= base;
	}

	return result;
}

void MomentumState(MDVector<float> &wavefunc, unsigned mode, float dx) {
	std::complex<float> exponent = 2if * (std::numbers::pi_v<float> * mode);
	for (std::size_t i = 0; i < wavefunc.size(); i += 2) {
		std::complex<float> exp = std::exp(exponent * (static_cast<float>(i) / wavefunc.size()));
		wavefunc[i] = exp.real();
		wavefunc[i+1] = exp.imag();
	}

	gpe::Normalize(wavefunc, dx);
}

void HarmonicBaseState(MDVector<float> &wavefunc, float omega, std::vector<float> dx) {
	float A = std::pow(omega / std::numbers::pi, .25f*dx.size());
	float exponent = -1 * omega / 2;
	
	std::vector<std::size_t> Nx = wavefunc.shape();
	Nx.back() /= 2;
	
	bool imag_flag = false; // Ignore imaginary guys
	auto indices = wavefunc.MDIndexGen();
	for (auto MDi : indices) {
		if (imag_flag) {
			imag_flag = false;
			wavefunc[MDi] = 0;
			continue;
		}

		MDi.back() /= 2;
		auto x_sqr = std::views::zip_transform([](std::size_t i, float d, std::size_t N)
		                                        { return IntegerPower(i*d - N*d/2, 2); },
							MDi, dx, Nx);
		float r_sqr = std::reduce(x_sqr.begin(), x_sqr.end(), .0f);
		float exp = std::exp(exponent * r_sqr);
		MDi.back() *= 2;
		wavefunc[MDi] = A*exp;
		imag_flag = true;
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
	float operator()(const std::vector<float> &x) {
		return 0.f;
	}
};

struct HarmonicPotential {
	float coeff = 0.5f; // In our units coeff=omega^2/2, omega=sqrt(2*coeff) base-state energy is sqrt(coeff/2)
	float operator()(const std::vector<float> &x) {
		float r_sqr = std::reduce(x.begin(), x.end(), 0.f,
		                          [](float acc, float val){ return acc + val*val; });
		return coeff*r_sqr;
	}
};

struct GaugedHarmonicPotential {
	float coeff = 0.5f;
	float gauge = 0.5f;
	float operator()(const std::vector<float> &x) {
		float r_sqr = std::reduce(x.begin(), x.end(), 1.0,
		                          [](float acc, float val){ return acc*val*val; });
		return coeff*r_sqr - gauge;
	}
};

int main(int argc, char **argv) {
	float space_length, space_step_ratio, g, time_step, soliton_distance;
	bool imag_time;
	std::size_t sim_duration;
	std::string config_filename = gpe::GetConfigFile(argc, argv);

	if (config_filename.at(0) == '\0') {
		std::print("{}", config_filename.substr(1));
		return 0;
	}

	gpe::ParamReader parser;
	parser.DefineParams("x_span", space_length, "g", g,
	                    "x_step_ratio", space_step_ratio, "sim_duration", sim_duration,
			    "time_step", time_step, "soliton_distance", soliton_distance,
			    "imag_time", imag_time);

	parser.ReadParams(std::ifstream(config_filename));

	float coeff = 0.5f;
	gpe::Simulation testsim = gpe::CreateGPESim({ space_length, space_length }, g,
	                                            { space_step_ratio, space_step_ratio }, time_step);
	if (imag_time) testsim.dt *= -1.0if;
	testsim.potential = NullPotential();
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(testsim.shape());
	HarmonicBaseState(wavefunc, std::sqrt(2*coeff), testsim.Getdx());
	// RandomState(wavefunc, testsim.Getdx(0));
	testsim.WriteWavefunc(wavefunc);
	
	testsim.Run(sim_duration);

	testsim.ReadWavefunc(wavefunc);
	std::print("Base state norm: {}\n", gpe::Norm(wavefunc, testsim.GetdV()));
	
	gpe::SaveWavefunc(wavefunc, "results/test.npy");
}
