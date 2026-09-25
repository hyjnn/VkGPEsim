#include "sim.hpp"
#include "MDVector.hpp"
#include <print>
#include <vector>

int main() {
	std::vector<std::size_t> sim_size = { 50 };
	gpe::Simulation testsim(sim_size);
	MDVector<float> test(sim_size);
	MDVector<float> wavefunc = gpe::CreateWavefuncVector<float>(sim_size);
	
	wavefunc[0] = 1;

	testsim.WriteWavefunc(wavefunc);
	testsim.QueueKick(1);
	testsim.Start();
	testsim.AwaitEnd();
	testsim.ReadWavefunc(wavefunc);

	for (std::size_t i = 0; i < wavefunc.shape[0]; i++) {
		std::print("{} ", wavefunc[i]);
	}
}
