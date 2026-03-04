cmake_minimum_required(VERSION 3.20)

project(bit_hamming_distance
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests" FORCE)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

FetchContent_MakeAvailable(benchmark)

add_library(hamming_distance_lib INTERFACE
    include/hamming_distance.hpp
)

target_include_directories(hamming_distance_lib
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_bit_hamming_distance
    benchmarks/bench_hamming_distance.cpp
)

target_link_libraries(benchmark_bit_hamming_distance
    PRIVATE
        hamming_distance_lib
        benchmark::benchmark
)
