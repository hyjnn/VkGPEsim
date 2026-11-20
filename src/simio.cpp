#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>

#ifdef HDF5_AVAILABLE
// Unfortunately hdf5 was not designed with IWYU in mind :(
#include "H5Cpp.h" // IWYU pragma: keep
#endif

#include "simio.hpp"
#include "MDVector.hpp"

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

		// Saves shape vector to numpy .npy format friendly string 
		std::string ShapeToNpy(std::vector<std::size_t> shape) {
			shape.back() /= 2;
			if (shape.size() == 0) return "(0,)";
			if (shape.size() == 1) return std::format("({},)", shape[0]);
			std::string output = std::format("({:d}", shape[0]);

			for (auto i = shape.begin() + 1; i != shape.end(); i++) {
				output += std::format(", {:d}", *i);
			}
			output += ")";

			return output;
		}

		bool SaveWavefunc_numpy(const MDVector<float> &src, std::ostream &outfile) {
			std::string shape = ShapeToNpy(src.shape());
			// For now assumes we are using 32-bit floats
			std::string header = "{'descr': '<c8',"
					     " 'fortran_order': True,"
					     " 'shape': " + shape + ", }";

			unsigned pad_len = (header.size() + 10) % 64;
			pad_len = 63 - pad_len;
			header += std::string(pad_len, '\x20'); // Pad with spaces
			header += '\n';
		
			std::array<char, 2> header_len_bytes = { static_cast<char>(header.size() & 0xff),
								 static_cast<char>((header.size() >> 8) & 0xff) };
			std::string preheader = std::string("\x93NUMPY\x01") + '\x00' +
			                        header_len_bytes[0] + header_len_bytes[1];
			outfile << preheader + header;	
			outfile.write(reinterpret_cast<const char *>(src.raw_data()), src.size() * sizeof(float));

			return true;
		}

#ifdef HDF5_AVAILABLE
		bool SaveWavefunc_hdf5(const MDVector<float> &src, std::string filename) {
			H5::H5File outfile(filename, H5F_ACC_TRUNC);
			auto shape = src.shape();
			shape.back() /= 2;
			H5::DataSpace dspace(shape.size(), shape.data());

			H5::CompType file_complex(2*sizeof(float));
			file_complex.insertMember("real", sizeof(float), H5::PredType::IEEE_F32LE);
			file_complex.insertMember("imag", sizeof(float), H5::PredType::IEEE_F32LE);

			H5::CompType mem_complex(2*sizeof(float));
			mem_complex.insertMember("real", sizeof(float), H5::PredType::NATIVE_FLOAT);
			mem_complex.insertMember("imag", sizeof(float), H5::PredType::NATIVE_FLOAT);

			H5::DataSet dset = outfile.createDataSet("wavefunction", file_complex, dspace);
			dset.write(src.raw_data(), mem_complex);

			return true;
		}
#endif
	}

	// Returns file name or an error message with byte \0 in front
	std::string GetConfigFile(int argc, char **argv) {
		if (argc != 2) {
			std::string error_msg = "Specify 1 positional argument to the program: "
						"path to the config file\n";
			return '\0' + error_msg;
		}
		if (!std::filesystem::exists(argv[argc-1])) {
			std::string error_msg = "Supplied file " + std::string(argv[argc-1]) +
						" does not exist\n";
			return '\0' + error_msg;
		}
		return argv[argc-1];
	}

	void ParamReader::ReadParams(std::istream &input) {
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

	bool SaveWavefunc(const MDVector<float> &src, std::string filename, bool overwrite) {
		std::ios::openmode mode = std::ios::binary;
		mode |= (overwrite ? std::ios::trunc : std::ios::noreplace);
		std::ofstream outfile(filename, mode);

		if (!outfile.is_open()) {
			std::print("Failed to open file {}\n", filename);
			return false;
		}

		std::size_t ext_separator = filename.rfind(".");
		std::string extension = filename.substr(ext_separator + 1);
		if (extension == "npy") {
			return SaveWavefunc_numpy(src, outfile);
		}
#ifdef HDF5_AVAILABLE
		else if (extension == "h5") {
			return SaveWavefunc_hdf5(src, filename);
		}
#endif
		else {
			std::print("Extension {} unrecognised\n", extension);
			return false;
		}
	}
}
