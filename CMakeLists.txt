cmake_minimum_required(VERSION 4.3)

project(GPEsim VERSION 0.1 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VKFFT_BACKEND 1 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 -, HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
set(VKFFT_MAX_FFT_DIMENSIONS 3)

add_library(MDVeclib INTERFACE)
target_include_directories(MDVeclib INTERFACE src/)
target_compile_features(MDVeclib INTERFACE cxx_std_23)

find_package(Vulkan REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(SPIRV-Tools REQUIRED)

add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=0)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/vkFFT/ /usr/include/glslang/Include)
target_link_libraries(VkFFT INTERFACE Vulkan::Vulkan glslang::glslang SPIRV-Tools)
target_compile_definitions(VkFFT INTERFACE -DVK_API_VERSION=12)#10 - Vulkan 1.0, 11 - Vulkan 1.1, 12 - Vulkan 1.2

add_library(utils_VkFFT STATIC src/utils_vkFFT/utils_VkFFT.cpp)
target_include_directories(utils_VkFFT PUBLIC src/utils_VkFFT)
target_link_libraries(utils_VkFFT PUBLIC VkFFT)

add_library(sim_backend STATIC src/sim.cpp)
target_include_directories(sim_backend PUBLIC src)
target_link_libraries(sim_backend PUBLIC MDVeclib)
target_compile_features(sim_backend PUBLIC cxx_std_23)
target_link_libraries(sim_backend PUBLIC VkFFT utils_VkFFT)

add_executable(GPEsim_exe src/main.cpp)
target_include_directories(GPEsim_exe PRIVATE src)
target_link_libraries(GPEsim_exe PRIVATE sim_backend)
target_compile_features(GPEsim_exe PRIVATE cxx_std_23)
set_property(TARGET GPEsim_exe PROPERTY OUTPUT_NAME GPEsim)

