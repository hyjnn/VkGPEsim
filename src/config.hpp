#include <cstddef>
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
			std::stringstream str;

			template<typename T>
			void operator()(T &variant_val) {
				str >> variant_val.get();
			}

			StringToVariant(std::string raw_str) : str(raw_str) {};
		};
		
		template<typename T, typename ...U>
		concept convertible_to_one_of = (std::is_convertible_v<T, U> || ...);
	}

	inline std::string GetConfigFile(int argc, char **argv) {
		if (argc != 2) {
			std::string error_msg = "Specify 1 positional argument to the program: path to the config file\n";
			return '\0' + error_msg;
		}
		else {
			return argv[argc-1];
		}
	}
	
	class ParamReader {
	public:
		using DestinationType = std::variant<std::reference_wrapper<std::size_t>,
		                                     std::reference_wrapper<float>,
						     std::reference_wrapper<std::string>>;
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
