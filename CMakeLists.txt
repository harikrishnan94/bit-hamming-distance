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

FetchContent_Declare(
    simde
    GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
    GIT_TAG v0.8.2
)

FetchContent_MakeAvailable(benchmark)

FetchContent_GetProperties(simde)
if(NOT simde_POPULATED)
    FetchContent_Populate(simde)
endif()

add_library(hamming_distance_lib
    src/hamming_distance.cpp
)

add_library(hamming_distance_lib_native
    src/hamming_distance.cpp
)

target_include_directories(hamming_distance_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${simde_SOURCE_DIR}
)

target_include_directories(hamming_distance_lib_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${simde_SOURCE_DIR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hamming_distance_lib_native PRIVATE -march=native)
elseif(MSVC)
    target_compile_options(hamming_distance_lib_native PRIVATE /arch:AVX2)
endif()

function(filter_out_debug_flags out_var)
    set(filtered_flags)
    foreach(flag IN LISTS ARGN)
        if(flag MATCHES "^SHELL:")
            string(REGEX REPLACE "^SHELL:" "" shell_flag_group "${flag}")
            separate_arguments(shell_flags UNIX_COMMAND "${shell_flag_group}")
            filter_out_debug_flags(filtered_shell_flags ${shell_flags})
            list(APPEND filtered_flags ${filtered_shell_flags})
            continue()
        endif()

        if(flag MATCHES " " AND NOT flag MATCHES "^\\$<")
            separate_arguments(split_flags UNIX_COMMAND "${flag}")
            filter_out_debug_flags(filtered_split_flags ${split_flags})
            list(APPEND filtered_flags ${filtered_split_flags})
            continue()
        endif()

        if(flag MATCHES "^-g.*$")
            continue()
        endif()
        if(flag MATCHES "^-f(debug|standalone-debug|no-standalone-debug|emit-class-debug-always|eliminate-unused-debug-types|no-eliminate-unused-debug-types).*$")
            continue()
        endif()
        if(flag MATCHES "^/Z(i|I|7)$")
            continue()
        endif()
        if(flag MATCHES "^/DEBUG(:.*)?$")
            continue()
        endif()
        if(flag MATCHES "^\\$<.*:(-g[^>]*|-f(debug|standalone-debug|no-standalone-debug|emit-class-debug-always|eliminate-unused-debug-types|no-eliminate-unused-debug-types)[^>]*|/Z(i|I|7)|/DEBUG(:[^>]*)?)>$")
            continue()
        endif()
        list(APPEND filtered_flags "${flag}")
    endforeach()
    set(${out_var} "${filtered_flags}" PARENT_SCOPE)
endfunction()

set(HAMMING_DISTANCE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/hamming_distance.cpp)
set(HAMMING_DISTANCE_ASM ${CMAKE_CURRENT_BINARY_DIR}/hamming_distance.s)

set(HAMMING_DISTANCE_COMPILER_FLAGS ${CMAKE_CXX_FLAGS})
if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" HAMMING_DISTANCE_BUILD_TYPE)
    list(APPEND HAMMING_DISTANCE_COMPILER_FLAGS ${CMAKE_CXX_FLAGS_${HAMMING_DISTANCE_BUILD_TYPE}})
endif()
separate_arguments(HAMMING_DISTANCE_COMPILER_FLAGS)
filter_out_debug_flags(HAMMING_DISTANCE_COMPILER_FLAGS ${HAMMING_DISTANCE_COMPILER_FLAGS})

get_target_property(HAMMING_DISTANCE_DEFINITIONS hamming_distance_lib COMPILE_DEFINITIONS)
if(HAMMING_DISTANCE_DEFINITIONS)
    foreach(definition IN LISTS HAMMING_DISTANCE_DEFINITIONS)
        list(APPEND HAMMING_DISTANCE_TARGET_FLAGS -D${definition})
    endforeach()
endif()

get_target_property(HAMMING_DISTANCE_OPTIONS hamming_distance_lib COMPILE_OPTIONS)
if(HAMMING_DISTANCE_OPTIONS)
    filter_out_debug_flags(HAMMING_DISTANCE_OPTIONS ${HAMMING_DISTANCE_OPTIONS})
    list(APPEND HAMMING_DISTANCE_TARGET_FLAGS ${HAMMING_DISTANCE_OPTIONS})
endif()

get_target_property(HAMMING_DISTANCE_INCLUDES hamming_distance_lib INCLUDE_DIRECTORIES)
if(HAMMING_DISTANCE_INCLUDES)
    foreach(include_dir IN LISTS HAMMING_DISTANCE_INCLUDES)
        list(APPEND HAMMING_DISTANCE_TARGET_FLAGS -I${include_dir})
    endforeach()
endif()

add_custom_command(
    OUTPUT ${HAMMING_DISTANCE_ASM}
    COMMAND ${CMAKE_CXX_COMPILER}
            -S
            $<$<BOOL:${CMAKE_CXX_STANDARD}>:-std=c++${CMAKE_CXX_STANDARD}>
            ${HAMMING_DISTANCE_TARGET_FLAGS}
            ${HAMMING_DISTANCE_COMPILER_FLAGS}
            -o ${HAMMING_DISTANCE_ASM}
            ${HAMMING_DISTANCE_SOURCE}
    DEPENDS ${HAMMING_DISTANCE_SOURCE}
    VERBATIM
    COMMAND_EXPAND_LISTS
)

add_custom_target(hamming_distance_asm ALL DEPENDS ${HAMMING_DISTANCE_ASM})

add_executable(benchmark_bit_hamming_distance
    benchmarks/bench_hamming_distance.cpp
)

add_executable(benchmark_bit_hamming_distance_native
    benchmarks/bench_hamming_distance.cpp
)

target_link_libraries(benchmark_bit_hamming_distance
    PRIVATE
        hamming_distance_lib
        benchmark::benchmark
)

target_link_libraries(benchmark_bit_hamming_distance_native
    PRIVATE
        hamming_distance_lib_native
        benchmark::benchmark
)

add_executable(simple_bench_bit_hamming_distance
    benchmarks/simple_bench_hamming_distance.cpp
)

add_executable(simple_bench_bit_hamming_distance_native
    benchmarks/simple_bench_hamming_distance.cpp
)

target_link_libraries(simple_bench_bit_hamming_distance
    PRIVATE hamming_distance_lib
)

target_link_libraries(simple_bench_bit_hamming_distance_native
    PRIVATE hamming_distance_lib_native
)
