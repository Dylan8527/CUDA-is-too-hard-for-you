cmake_minimum_required(VERSION 3.12)
project(CudaTemplate VERSION 1.0.0 LANGUAGES C CXX CUDA DESCRIPTION "A cuda template project")

##--------------------- Version file -------------------------------------##
configure_file(src/c/cuda_template_version.h.in "src/c/cuda_template_version.h")

# Specify the language standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
find_pa

##--------------------- Compile options ----------------------------------##
# Configure compile options 
add_compile_options(-Wall -Wextra -pedantic -Werror)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++11")

# Set build type
if (NOT CMAKE_BUILD_TYPE)
    message("Use 'cmake -DCMAKE_BUILD_TYPE=Debug' better")
    set(CMAKE_BUILD_TYPE Debug)     # Release, RelWithDebInfo, Debug
                                    # Use 'cmake -DCMAKE_BUILD_TYPE=Debug' better
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compile options for Debug variant
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Compile options for Release variant 
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

message(STATUS "Compile options for c: ${CMAKE_C_FLAGS}")
message(STATUS "Compile options for c++: ${CMAKE_CXX_FLAGS}")

##--------------------- Global Macros ------------------------------------##


##--------------------- Include directories ------------------------------##


##--------------------- Source files -------------------------------------##


##--------------------- Build executable ---------------------------------##


##--------------------- Install headers ---------------------------------##
