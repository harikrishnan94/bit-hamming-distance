#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>

#include "hamming_distance.hpp"

namespace {
void BM_HammingDistance(benchmark::State& state) {
    constexpr std::array<std::uint64_t, 8> lhs_values{
        0ULL,
        1ULL,
        0xAAAAAAAAAAAAAAAAULL,
        0x5555555555555555ULL,
        0x0123456789ABCDEFULL,
        0xFEDCBA9876543210ULL,
        0xFFFFFFFF00000000ULL,
        0x00000000FFFFFFFFULL
    };

    constexpr std::array<std::uint64_t, 8> rhs_values{
        0ULL,
        2ULL,
        0x5555555555555555ULL,
        0xAAAAAAAAAAAAAAAAULL,
        0x0011223344556677ULL,
        0x8899AABBCCDDEEFFULL,
        0x00000000FFFFFFFFULL,
        0xFFFFFFFF00000000ULL
    };

    std::size_t idx = 0;
    for (auto _ : state) {
        auto result = hamming_distance(lhs_values[idx], rhs_values[idx]);
        benchmark::DoNotOptimize(result);
        idx = (idx + 1) % lhs_values.size();
    }
}
}  // namespace

BENCHMARK(BM_HammingDistance);
BENCHMARK_MAIN();
