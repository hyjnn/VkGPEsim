#include "MDVector.hpp"
#include <cstddef>
#include <functional>
#include <istream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace gpe {
	namespace {
		template<typename T, typename ...U>
		concept convertible_to_one_of = (std::is_convertible_v<T, U> || ...);

	}

	std::string GetConfigFile(int argc, char **argv);
	
	class ParamReader {
	public:
		using DestinationType = std::variant<std::reference_wrapper<std::size_t>,
		                                     std::reference_wrapper<float>,
						     std::reference_wrapper<std::string>,
		                                     std::reference_wrapper<bool>>;
		using Parameter = std::pair<std::string, DestinationType>;

		template<typename T1, typename T2, typename ...U>
		requires (std::is_convertible_v<T1, std::string> &&
		          convertible_to_one_of<T2, std::size_t, float, std::string>)
		void DefineParams(const T1 &name, T2& destination, U &...rest) {
			param_map.emplace(name, destination);
			DefineParams(rest...);
		}

		void DefineParams() {}
		
		void ReadParams(std::istream &input);

		void ReadParams(std::istream &&input) {
			ReadParams(input);
		}
	private:
		std::unordered_map<std::string, DestinationType> param_map;
	};

	// Saves src to file named filename. If overwrite is set to false, refuses to overwrite existing file.
	// Format is deduced based on the filename, currently supported: .npy (numpy), .h5 (hdf5)
	bool SaveWavefunc(const MDVector<float> &src, std::string filename, bool overwrite = true);
}
