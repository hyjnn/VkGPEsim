#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <istream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace gpe {
	namespace {
		struct StringToVariant {
			std::string str;

			template<typename T>
			void operator()(T &variant_val) {
				std::stringstream(str) >> variant_val.get();
			}

			void operator()(std::reference_wrapper<bool> &variant_bool) {
				std::ranges::transform(str, str.begin(), [](char c){ return std::tolower(c); });
				if (str == "0" || str == "false") {
					variant_bool.get() = false;
				}
				else if (str == "1" || str == "true") {
					variant_bool.get() = true;
				}
			}

			StringToVariant(std::string str) : str(str) {};
		};
		
		template<typename T, typename ...U>
		concept convertible_to_one_of = (std::is_convertible_v<T, U> || ...);
	}

	// Returns file name or an error message with byte \0 in front
	inline std::string GetConfigFile(int argc, char **argv) {
		if (argc != 2) {
			std::string error_msg = "Specify 1 positional argument to the program: path to the config file\n";
			return '\0' + error_msg;
		}
		if (!std::filesystem::exists(argv[argc-1])) {
			std::string error_msg = "Supplied file " + std::string(argv[argc-1]) + " does not exist\n";
			return '\0' + error_msg;
		}
		return argv[argc-1];
	}
	
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
		
		void ReadParams(std::istream &input) {
			std::string line, param_name, param_value;

			while (std::getline(input, line)) {
				std::erase_if(line, [](char c){ return std::isspace(c); }); // Ignore all whitespace
				if (line.starts_with('#') || line.starts_with("//")) continue; // Ignore comments

				std::size_t delimiter = line.find("=");
				param_name = line.substr(0, delimiter);

				if (param_map.contains(param_name)) {
					param_value = line.substr(delimiter + 1);
					std::visit(StringToVariant(param_value), param_map.at(param_name));
				}
			}
		}

		void ReadParams(std::istream &&input) {
			ReadParams(input);
		}
	private:
		std::unordered_map<std::string, DestinationType> param_map;
	};
}
